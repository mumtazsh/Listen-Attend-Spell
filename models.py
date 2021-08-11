import torch
import torch.nn as nn
import torch.nn.utils as utils
from random import random
from torchnlp.nn import LockedDropout
from weight_drop import WeightDrop

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        
        #print(key.size())
        #print(query.size())
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(DEVICE)
        energy.masked_fill_(mask, -1e9)
        attention = nn.functional.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        return context, attention
        

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x):
        
        x, x_lens = utils.rnn.pad_packed_sequence(x)
        #print(x.size())
        x = x.transpose(0,1)
        if (x.size(1) % 2) != 0:
            x = x[:,:-1,:]
        x = torch.reshape(x, (x.size(0),x.size(1)//2, x.size(2)*2))
        x = x.transpose(0,1)
        x = utils.rnn.pack_padded_sequence(x, lengths=x_lens//2, batch_first=False, enforce_sorted=False)
        return self.blstm(x)[0]

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        
        
        self.pblstm1 = pBLSTM(4*hidden_dim, hidden_dim)
        self.pblstm2 = pBLSTM(4*hidden_dim, hidden_dim)
        self.pblstm3 = pBLSTM(4*hidden_dim, hidden_dim)

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        
        outputs, _ = self.lstm(rnn_inp)
        
        
        
        outputs = self.pblstm1(outputs)
        outputs = self.pblstm2(outputs)
        outputs = self.pblstm3(outputs)

        linear_input, out_lens = utils.rnn.pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, out_lens


class Decoder(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, isTrain=True):
        
        batch_size = key.shape[1]

        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = (torch.ones(batch_size, 1, dtype=torch.int)*33).to(DEVICE)#(torch.ones(batch_size,35)*-float("Inf")).to(DEVICE)##
        #prediction[:,33] = 0
        #print(max_len)
        all_attentions=[]
        context = values[0,:,:]
        for i in range(max_len):
             

            if (isTrain):
                if random() <= 0.9 or i == 0:
                    char_embed = embeddings[:,i,:]
                else:
                    #print(i, " ", prediction.dtype)
                    temp = nn.functional.gumbel_softmax(prediction)
                    temp2 = torch.multinomial(temp,1)
                    char_embed = torch.squeeze(self.embedding(temp2))
            else:
                #print(prediction.dtype)
                char_embed = torch.squeeze(self.embedding(prediction)) #self.embedding(prediction.argmax(dim=-1))
                

            if (self.isAttended == True):
                inp = torch.cat([char_embed, context], dim=1)
            else:
                inp = torch.cat([char_embed, values[i,:,:]], dim=1)
            
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]
            if self.isAttended:
                context, att_mask = self.attention(output, key.transpose(0,1), values.transpose(0,1), lens)
                prediction = self.character_prob(torch.cat([output, context], dim=1))
                all_attentions.append(att_mask.detach())
            else:
                prediction = self.character_prob(torch.cat([output, values[i,:,:]], dim=1))
            
            if (isTrain):
                predictions.append(prediction.unsqueeze(1))
            else:
                #m = torch.nn.Softmax(dim=1)
                #prediction = torch.multinomial(m(prediction), 1)
                prediction = torch.multinomial(nn.functional.gumbel_softmax(prediction), 1)
                predictions.append(prediction.unsqueeze(1))
        #print(len(predictions))
        
        return torch.cat(predictions, dim=1), torch.stack(all_attentions, dim=1)


class Seq2Seq(nn.Module):
    
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, value_size, key_size)
        self.decoder = Decoder(vocab_size, 2*hidden_dim, value_size, key_size, isAttended=isAttended)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        key, value, out_lens = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions, attentions = self.decoder(key, value, out_lens, text_input)
        else:
            predictions, attentions = self.decoder(key, value, out_lens, text=None, isTrain=False)
        return predictions, attentions



    

class PackedLanguageModel(nn.Module):
    
    def __init__(self,vocab_size,embed_size,hidden_size, nlayers, stop):
        super(PackedLanguageModel,self).__init__()
        self.vocab_size=vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers=nlayers
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers) # 1 layer, batch_size = False
        self.scoring = nn.Linear(hidden_size,vocab_size)
        self.stop = stop # stop line character (\n)
    
    def forward(self,seq_list): # list
        batch_size = len(seq_list)
        lens = [len(s) for s in seq_list] # lens of all lines (already sorted)
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1]+l) 
        seq_concat = torch.cat(seq_list) # concatenated sequence
        embed_concat = self.embedding(seq_concat) # concatenated embeddings
        embed_list = [embed_concat[bounds[i]:bounds[i+1]] for i in range(batch_size)] 
        packed_input = utils.rnn.pack_sequence(embed_list, enforce_sorted=False) 
        
        
        
        
        
        hidden = None
        output_packed,hidden = self.rnn(packed_input,hidden)
        output_padded, _ = utils.rnn.pad_packed_sequence(output_packed) 
        output_flatten = torch.cat([output_padded[:lens[i],i] for i in range(batch_size)]) # concatenated output
        scores_flatten = self.scoring(output_flatten) # concatenated logits
        return scores_flatten # return concatenated logits
    
    def generate(self,seq, n_words): # L x V
        generated_words = []
        embed = self.embedding(seq).unsqueeze(1) # L x 1 x E
        hidden = None
        output_lstm, hidden = self.rnn(embed,hidden) # L x 1 x H
        output = output_lstm[-1] # 1 x H
        scores = self.scoring(output) # 1 x V
        _,current_word = torch.max(scores,dim=1) # 1 x 1
        generated_words.append(current_word)
        if n_words > 1:
            for i in range(n_words-1):
                embed = self.embedding(current_word).unsqueeze(0) # 1 x 1 x E
                output_lstm, hidden = self.rnn(embed,hidden) # 1 x 1 x H
                output = output_lstm[0] # 1 x H
                scores = self.scoring(output) # V
                _,current_word = torch.max(scores,dim=1) # 1
                generated_words.append(current_word)
                if current_word[0].item()==self.stop: # If end of line
                    break
        return torch.cat(generated_words,dim=0)
    
    
class LanguageModel(nn.Module):
    
    def __init__(self, vocab_size, embed_size,hidden_size, nlayers):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size #400
        self.hidden_size = hidden_size #1150
        self.nlayers = nlayers #3
        self.lockdrop = LockedDropout(p=0.65)
        self.lockdrop2 = LockedDropout(p=0.2)
        #self.idrop = nn.Dropout(p=0.65)
        #self.hdrop = nn.Dropout(p=0.2)
        self.lockdrop3 = LockedDropout(p=0.4)
        self.embedding = nn.Embedding(vocab_size,self.embed_size)
        self.rnns = [torch.nn.LSTM(self.embed_size if l == 0 else self.hidden_size, self.hidden_size, 1, dropout=0) for l in range(self.nlayers)]
        self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=0.5) for rnn in self.rnns]
        #self.rnns = [WeightDropLSTM(self.embed_size if l == 0 else self.hidden_size, self.hidden_size, 1, dropout=0, weight_dropout=0.5) for l in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(self.rnns)
        #self.rnn = WeightDropLSTM(input_size = self.embed_size,hidden_size=self.hidden_size,num_layers=self.nlayers, dropout=0.3, weight_dropout=0.5)
        self.scoring = nn.Linear(self.hidden_size,vocab_size)
        self.init_weights()
        

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.scoring.bias.data.fill_(0)
        self.scoring.weight.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        return [(weight.new(1, bsz, self.hidden_size).zero_(), weight.new(1, bsz, self.hidden_size).zero_()) for l in range(self.nlayers)]
            
    def forward(self, x, hidden):
        
        batch_size = x.size(1)
        #print(batch_size)
        #x1 = self.dp1(x).long()
        embed = self.embedding(x) #L x N x E
        embed = self.lockdrop(embed)
        
        raw_output = embed
        new_hidden = []
        #raw_outputs = []
        #outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
         #   raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop2(raw_output)
          #      outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop3(raw_output)
        #outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        output_flatten = self.scoring(result)
        #print("Mod",output_flatten.size())
        return output_flatten.view(-1,batch_size,self.vocab_size), hidden
        
        #hidden = None
        #output_lstm,hidden = self.rnn(embed,hidden) #L x N x H
        #output_lstm = self.dp3(output_lstm)
        #output_lstm_flatten = output_lstm.view(-1,self.hidden_size) #(L*N) x H
        #output_flatten = self.scoring(output_lstm_flatten) #(L*N) x V
        #return output_flatten
        #return output_flatten.view(-1,batch_size,self.vocab_size)
        
    
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors,
        to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    def generate(self,seq, n_words): # L x V
        # performs greedy search to extract and return words (one sequence).
        generated_words = []
        hidden = self.init_hidden(seq.size(1))
        embed = self.embedding(seq) # L x N x E
               
        raw_output = embed
        new_hidden = []
        hidden = self.repackage_hidden(hidden)
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            
        hidden = new_hidden

             
        
        #output_lstm, hidden = self.rnn(embed,hidden) # L x N x H
        output = raw_output[-1] # N x H
        scores = self.scoring(output) # N x V
        _,current_word = torch.max(scores,dim=1) # N x 1
        generated_words.append(current_word.unsqueeze(1)) #??
        if n_words > 1:
            for i in range(n_words-1):
                #embed = self.embedding(current_word).unsqueeze(0) # 1 x N x E
                #output_lstm, hidden = self.rnn(embed,hidden) # 1 x N x H
                embed = self.embedding(current_word).unsqueeze(0) # L x N x E
               
                raw_output = embed
                new_hidden = []
                hidden = self.repackage_hidden(hidden)
                for l, rnn in enumerate(self.rnns):
                    current_input = raw_output
                    raw_output, new_h = rnn(raw_output, hidden[l])
                    new_hidden.append(new_h)
            
                hidden = new_hidden
                output = raw_output[0] # N x H
                scores = self.scoring(output) # N x V
                _,current_word = torch.max(scores,dim=1) # N
                generated_words.append(current_word.unsqueeze(1)) #??
        return torch.cat(generated_words,dim=-1)
