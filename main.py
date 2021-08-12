import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models import Seq2Seq, PackedLanguageModel, LanguageModel
from train_test import train, test, val, train_epoch_packed, LanguageModelTrainer
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset, LinesDataset, collate_lines, LanguageModelDataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=256, isAttended=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(reduction='none')
    nepochs = 20
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)
    
    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

      
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)
    
    lang_model = PackedLanguageModel(len(LETTER_LIST),512,512,3, stop=34)

    #Alternate Language Model - Training
    """
    lang_model = LanguageModel(len(LETTER_LIST),400,1150,3)
    lm_train_loader = LanguageModelDataLoader(dataset=character_text_train, batch_size=80, shuffle=True)
    lm_val_loader = LanguageModelDataLoader(dataset=character_text_valid, batch_size=80, shuffle=False)
    lm_optimizer = torch.optim.SGD(lang_model.parameters(), lr=30, weight_decay=1.2e-6)
    trainer = LanguageModelTrainer(model=lang_model, loader=lm_train_loader, val_loader=lm_val_loader, optimizer = lm_optimizer, max_epochs=20)
    
    lang_model = lang_model.to(DEVICE)
    best_lm_perp = 5 
    for epoch in range(20):
        trainer.train()
        val_lpw = trainer.test()
        perp_lm_v = np.exp(val_lpw)
        print("Validation Perplexity: ", perp_lm_v)
        if perp_lm_v < best_lm_perp:
            best_lm_perp = perp_lm_v
            #print("Saving model, predictions and generated output for epoch "+str(epoch)+" with NLL: "+ str(best_nll))
            torch.save(lang_model.state_dict(), './best-val-lang-model-weight-drop.pt')
    """
    
    #Seq2Seq - Training
    
    
    best_perp = 1.75
    model.load_state_dict(torch.load('./best-val-new.pt'))
    for epoch in tqdm(range(nepochs)):
        train(model, train_loader, criterion, optimizer, epoch)
        perp_v = val(model, val_loader, criterion, epoch)
        if perp_v < best_perp:
            best_perp = perp_v
            torch.save(model.state_dict(), './best-val-new-2.pt')
    
    #model.load_state_dict(torch.load('./exp-1.pt'))
    #torch.save(model.state_dict(), './final-model-new.pt')
    
    
    #Language Model - Training
    
    lang_model = lang_model.to(DEVICE)
    
    lm_optimizer = torch.optim.Adam(lang_model.parameters(),lr=0.0001, weight_decay=1e-6)
    
    lm_train_dataset = LinesDataset(character_text_train)
    lm_val_dataset = LinesDataset(character_text_valid)
    lm_train_loader = DataLoader(lm_train_dataset, shuffle=True, batch_size=64, collate_fn = collate_lines)
    lm_val_loader = DataLoader(lm_val_dataset, shuffle=False, batch_size=64, collate_fn = collate_lines, drop_last=True)
    
    best_lm_perp = 3.35
    lang_model.load_state_dict(torch.load('./best-val-lang-model.pt'))
    for i in range(20):
        val_lpw = train_epoch_packed(lang_model, lm_optimizer, lm_train_loader, lm_val_loader)
        perp_lm_v = np.exp(val_lpw)
        if perp_lm_v < best_lm_perp:
            best_lm_perp = perp_lm_v
            torch.save(lang_model.state_dict(), './best-val-lang-model-2.pt')
    
    #torch.save(lang_model.state_dict(), "./trained_lang_model_2.pt")
    
    
    model.load_state_dict(torch.load('./best-val-new-2.pt'))
    lang_model.load_state_dict(torch.load('./best-val-lang-model-2.pt'))
    
    test(model, lang_model, test_loader, 20)
    
    

if __name__ == '__main__':
    main()
