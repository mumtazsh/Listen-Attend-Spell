import time
import torch
import numpy as np
from tests import test_prediction, test_generation
import pandas as pd
import os
from util import plot_attn_flow
### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(DEVICE)
    start = time.time()
    train_loss = 0
    batch_id = 0
    f = open("perp.txt","w")
    # 1) Iterate through your loader
    for batch_idx, (X, X_lens, Y, Y_lens) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        # 3) Set the inputs to the device.
        X = X.to(DEVICE)
        Y = Y.to(DEVICE) # all data & model on same device
        #X_lens = X_lens.to(DEVICE)
        #Y_lens = Y_lens.to(DEVICE)
        # 4) Pass your inputs, and length of speech into the model.
        #print(Y.size())
        out, att = model(X, X_lens, Y)
        # 5) Generate a mask based on the lengths of the text to create a masked loss.
        Y_lens = torch.unsqueeze(Y_lens,0)
        mask = torch.tensor(np.arange(Y.size(1)-1)).unsqueeze(1) < Y_lens
        # 5.1) Ensure the mask is on the device and is the correct shape.
        mask = mask.to(DEVICE)
        # 6) If necessary, reshape your predictions and origianl text input 
        # 6.1) Use .contiguous() if you need to. 
        
        # 7) Use the criterion to get the loss.
        #print(out.size())
        #print(Y.size())
        loss = criterion(out.transpose(1,2)[:,:,:-1], Y[:,1:])
        #print(loss.item())
        #running_loss += loss.item()
        #print("The batch is ",batch_idx)
        # 8) Use the mask to calculate a masked loss. 
        masked_loss = torch.sum(mask.transpose(0,1)*loss)
        #print(mask.size())
        #print(loss.size())
        #print(loss.dtype)
        # 9) Run the backward pass on the masked loss. 

        masked_loss.backward()
        # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        # 11) Take a step with your optimizer
        optimizer.step()
        # 12) Normalize the masked loss
        
        masked_loss /= torch.sum(Y_lens)
        # 13) Optionally print the training loss after every N batches
        train_loss += masked_loss.item()
        batch_id+=1
        if not os.path.exists('./experiments'):
            os.mkdir('./experiments')
        if batch_idx % 50 == 0:
            path=os.path.join('experiments', 'attentions-{}-{}.png'.format(epoch,batch_id))
            #print(Y_lens.size())
            #print(X_lens.size())
            plot_attn_flow(att[0,:Y_lens.squeeze()[0] - 1, :X_lens[0]].cpu(), path)
            with open('perp.txt', 'a') as f:
                print(epoch, " ", batch_idx, " Training perplexity :",np.exp(masked_loss.item()), file=f)
    train_lpw = train_loss / batch_id
    print("Training perplexity :",np.exp(train_lpw))
    end = time.time()

def val(model, val_loader, criterion, epoch):
    
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        start = time.time()
        val_loss = 0
        batch_id = 0
        for batch_idx, (X, X_lens, Y, Y_lens) in enumerate(val_loader):   
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            prediction,_ = model(X, X_lens, Y)
            #prediction = torch.squeeze(prediction)
            
            #batch_size = prediction.size(0)
            #char1 = (torch.ones(batch_size, 1, dtype=torch.int)*33).to(DEVICE)
            #prediction = torch.cat([char1, prediction], dim=1)
                
            Y_lens = torch.unsqueeze(Y_lens,0)
            mask = torch.tensor(np.arange(Y.size(1)-1)).unsqueeze(1) < Y_lens
        
            mask = mask.to(DEVICE)
        
        
            loss = criterion(prediction.transpose(1,2)[:,:,:-1], Y[:,1:])
        
            masked_loss = torch.sum(mask.transpose(0,1)*loss)
        
        
            masked_loss /= torch.sum(Y_lens)
            val_loss += masked_loss.item()
            batch_id+=1
        
        val_lpw = val_loss/batch_id
        
        print("Validation perplexity :",np.exp(val_lpw))
            
            
        end = time.time()
    return np.exp(val_lpw)
    
def test(model, lang_model, test_loader, epoch):
    ### Write your test code here! ###
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        lang_model.eval()
        lang_model.to(DEVICE)
        
        N = 1024
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        lm_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        Id = []
        samp_id = 0
        Predictions=[]
    
        for batch_idx, (X, X_lens) in enumerate(test_loader):   
            start = time.time()
            X = X.to(DEVICE)
            print(batch_idx)
            preds = []
            seq_lens = []
            scores = []
            lm_scores = []
            for i in range(N):
                prediction, _ = model(X, X_lens, isTrain=False)
                prediction = torch.squeeze(prediction)
            #print(prediction.size())
                batch_size = prediction.size(0)
                char1 = (torch.ones(batch_size, 1, dtype=torch.int)*33).to(DEVICE)
                prediction = torch.cat([char1, prediction], dim=1)
                preds.append(prediction)
            #print(prediction.size())
            
                seq_len = torch.zeros(batch_size, dtype=torch.int).to(DEVICE)
                seq_list = []
                targets = []
            #print(seq_len.size())
                for j in range(batch_size):
                #print((prediction[j] == 34).nonzero(as_tuple=True)[0])
                    temp = (prediction[j] == 34)
                    if torch.any(temp):
                        seq_len[j] = (prediction[j] == 34).nonzero(as_tuple=True)[0][0] + 1
                    else:
                        seq_len[j] = prediction.size(1)
                    seq_list.append(prediction[j,:seq_len[j]-1])
                    targets.append(prediction[j,1:seq_len[j]])
            #print(seq_len)
            #print(seq_len.size())
                seq_lens.append(seq_len)
            
                out, _ = model(X, X_lens, prediction)
                mask = torch.tensor(np.arange(prediction.size(1)-1)).to(DEVICE).unsqueeze(1) < seq_len
                mask = mask.to(DEVICE)
                loss = criterion(out.transpose(1,2)[:,:,:-1], prediction[:,1:])
                masked_loss = torch.sum(mask.transpose(0,1)*loss, dim=1)
                score = -1*masked_loss/seq_len
                
                lm_outputs = lang_model(seq_list)
                lm_loss_cum = lm_criterion(lm_outputs,torch.cat(targets)) # criterion of the concatenated output 
                lm_loss = torch.zeros(batch_size, dtype=torch.float32).to(DEVICE)
                counter  = 0
                for j in range(batch_size):
                    lm_loss[j] = torch.sum(lm_loss_cum[counter:counter+seq_len[j]-1])
                    counter += seq_len[j]-1
                
                lm_score = -1*lm_loss
                #lamb = 1.6
                #score += lamb*lm_score
                #print(masked_loss)
                #print(seq_len)
                #print(score)
                scores.append(score)
                lm_scores.append(lm_score)
            scores_t = torch.stack(scores, dim=1)
            lm_scores_t = torch.stack(lm_scores, dim=1)
            #print(scores_t.size())
            #print(seq_lens)
            best_loss_ind = scores_t.argmax(dim=1)
            k = 32
            _ , best32_loss_ind = torch.topk(scores_t, k, dim=1)
            lamb = 0.008
            scores_t += lamb*lm_scores_t
            #print(best_loss_ind)
            preds_t = torch.stack(preds, dim=2)
        #print(preds_t.size())
            best_preds = []
            for i in range(batch_size):
                curr_max = float('-inf')
                for j in range(k):
                    if scores_t[i,best32_loss_ind[i,j]] > curr_max:
                        curr_max = scores_t[i,best32_loss_ind[i,j]]
                        best_loss_ind[i] = best32_loss_ind[i,j]
                best_preds.append(torch.unsqueeze(preds_t[i,:,best_loss_ind[i]], 0))
        #print(best_preds[0].size())
            best_preds_t = torch.cat(best_preds, dim=0)
        #print(best_preds_t.size())
            seq_lens_t = torch.stack(seq_lens, dim=1)
            best_seq_lens = []
            for i in range(batch_size):
                best_seq_lens.append(seq_lens_t[i,best_loss_ind[i]])
        #print(best_preds[0].size())
            best_seq_lens_t = torch.tensor(best_seq_lens)
            #print(best_seq_lens_t)
            for i in range(batch_size):
                best_seq = best_preds_t[i, 1:best_seq_lens_t[i]-1]
                best_pron = ''.join(LETTER_LIST[i] for i in best_seq)
                #print(best_pron)
                Id.append(samp_id)
                Predictions.append(best_pron)
                samp_id += 1
            
            end = time.time()
            print("Time for this batch", end-start)
    data = {'Id': Id, 'label':Predictions }
    df = pd.DataFrame(data)
    df.to_csv("submission_lm_2.csv", header=True, index=False)
    #pass
    
  
    
def train_epoch_packed(model, optimizer, train_loader, val_loader):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum") # sum instead of averaging, to take into account the different lengths
    criterion = criterion.to(DEVICE)
    batch_id=0
    before = time.time()
    print("Training", len(train_loader), "number of batches")
    model.train()
    for inputs,targets in train_loader: # lists, presorted, preloaded on GPU
        batch_id+=1
        #print(inputs)
        #print(targets)
        outputs = model(inputs)
        loss = criterion(outputs,torch.cat(targets)) # criterion of the concatenated output
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            after = time.time()
            nwords = np.sum(np.array([len(l) for l in inputs]))
            lpw = loss.item() / nwords
            print("Time elapsed: ", after - before)
            print("At batch",batch_id)
            print("Training loss per word:",lpw)
            print("Training perplexity :",np.exp(lpw))
            before = after
    model.eval()
    val_loss = 0
    batch_id=0
    nwords = 0
    for inputs,targets in val_loader:
        nwords += np.sum(np.array([len(l) for l in inputs]))
        batch_id+=1
        outputs = model(inputs)
        loss = criterion(outputs,torch.cat(targets))
        val_loss+=loss.item()
    val_lpw = val_loss / nwords
    print("\nValidation loss per word:",val_lpw)
    print("Validation perplexity :",np.exp(val_lpw),"\n")
    return val_lpw

# model trainer
from tqdm import tqdm
class LanguageModelTrainer:
    def __init__(self, model, loader, val_loader, optimizer, max_epochs=1):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = optimizer #torch.optim.SGD(model.parameters(), lr=30, weight_decay=1.2e-6)
        #self.criterion = nn.NLLLoss()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def train(self):
         # set to training mode
        epoch_loss = 0
        num_batches = 0
        hidden = self.model.init_hidden(80)
        for batch_num, (inputs, targets) in enumerate(self.loader):
            #print(inputs)
            #print(inputs.size())
            #print(inputs.dtype)
            #print(targets.size())
            #print(targets.dtype)
            #print(hidden.size())
            #print(hidden.dtype)
            #print(len(inputs), len(targets), len(hidden))
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            #print(batch_num)
            raw_loss, hidden = self.train_batch(inputs, targets, hidden)
            if batch_num % 100 == 0:
                print("At batch",batch_num)
                print("Training loss per word:",raw_loss)
                print("Training perplexity :",np.exp(raw_loss))
            epoch_loss += raw_loss
    
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets, hidden):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        self.model.train()
        hidden = self.model.repackage_hidden(hidden)
        
        self.optimizer.zero_grad()
        outputs, new_hidden = self.model(inputs, hidden) # 3D
        #m = torch.nn.LogSoftmax(dim=1)
        #print(outputs.size())
        #print(targets.size())
        loss = self.criterion(outputs.view(-1,outputs.size(2)), targets.reshape(-1)) # Loss of the flattened outputs
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        #print(loss.item())
        return loss.item(), new_hidden
        #raise NotImplemented

    def val_batch(self, inputs, targets, hidden):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        self.model.eval()
        hidden = self.model.repackage_hidden(hidden)
        
        outputs, new_hidden = self.model(inputs, hidden) # 3D
        #m = nn.LogSoftmax(dim=1)
        #print(outputs.size())
        #print(targets.size())
        loss = self.criterion(outputs.view(-1,outputs.size(2)), targets.reshape(-1)) # Loss of the flattened outputs
               
        #print(loss.item())
        return loss.item(), new_hidden
    
    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        """
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        
        self.predictions.append(predictions)
        
        
        nll = test_prediction(predictions, fixtures_pred['out'])
        
        
        """
        epoch_loss = 0
        num_batches = 0
        hidden = self.model.init_hidden(80)
        for batch_num, (inputs, targets) in enumerate(self.val_loader):
                       
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            #print(batch_num)
            raw_loss, hidden = self.val_batch(inputs, targets, hidden)
            epoch_loss += raw_loss
    
        epoch_loss = epoch_loss / (batch_num + 1)
        #self.epochs += 1
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.val_losses.append(epoch_loss)
        
       
            
        #print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
         #             % (self.epochs + 1, self.max_epochs, nll))
        return epoch_loss

    
class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here
            
            :param inp:
            :return: a np.ndarray of logits
        """
        inp = torch.tensor(inp).long()
        inp = inp.transpose(0,1)
        
        hidden = model.init_hidden(inp.shape[1])
        hidden = repackage_hidden(hidden)
        out = model(inp, hidden)
       
        return out[0][-1].detach().numpy()
        #raise NotImplemented

        
    