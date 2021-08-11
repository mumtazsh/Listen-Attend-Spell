import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn.utils.rnn as rnn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'
'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid



def transform_letter_to_index(transcript, letter_list):
    
    out = []
    for utt in transcript: #sentence
        temp=[]
        temp.append(letter_list.index('<sos>'))
        for word in utt: #word
            for ch in word.decode("utf-8"): #character
                temp.append(letter_list.index(ch))
            temp.append(letter_list.index(' '))
        temp.append(letter_list.index('<eos>'))    
        out.append(np.array(temp))
    #print(len(out))
    return np.array(out, dtype=object)

class Speech2TextDataset(Dataset):
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    inputs = rnn.pad_sequence([s[0] for s in batch_data], padding_value=0.0)
    input_sizes = torch.LongTensor([s[0].shape[0] for s in batch_data])
    targets = rnn.pad_sequence([s[1] for s in batch_data],batch_first=True, padding_value=0)
    target_sizes = torch.LongTensor([s[1].shape[0] for s in batch_data])
    return inputs,input_sizes,targets, target_sizes
    #pass 


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    inputs = rnn.pad_sequence([s for s in batch_data], padding_value=0.0)
    input_sizes = torch.LongTensor([s.shape[0] for s in batch_data])
    return inputs,input_sizes
    #pass 
    

    
class LinesDataset(Dataset):
    def __init__(self,lines):
        self.lines=[torch.tensor(l) for l in lines]
    def __getitem__(self,i):
        line = self.lines[i]
        return line[:-1].to(DEVICE),line[1:].to(DEVICE)
    def __len__(self):
        return len(self.lines)



def collate_lines(seq_list):
    inputs,targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs,targets

class LanguageModelDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seq_len = 71

    def __iter__(self):
        
        if self.shuffle == True:
            np.random.shuffle(self.dataset)
        text = np.concatenate(self.dataset)
        n_seq = len(text) // self.seq_len
        text = text[:n_seq * self.seq_len]
        data = torch.tensor(text).view(-1,self.seq_len).long()
        number_of_batches = data.shape[0] // self.batch_size
        for i in range(number_of_batches):
            yield data[i*self.batch_size:(i+1)*self.batch_size,:-1].transpose(0,1), data[i*self.batch_size:(i+1)*self.batch_size,1:].transpose(0,1)
        
