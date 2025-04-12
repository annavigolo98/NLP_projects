from datetime import datetime
import numpy as np
from pydantic import BaseModel
import torch
from tqdm import tqdm



class Train(BaseModel):
    def train(self,
             model,
             criterion,
             optimizer, 
             train_loader, 
             valid_loader, 
             epochs, 
             device,
             tokenizer):
        
        train_losses = np.zeros(epochs)
        test_losses = np.zeros(epochs)

        for it in range(epochs):
            model.train()
            t0 = datetime.now()
            train_loss = []
            for batch in tqdm(train_loader):
                #Move data to the GPU
                batch = {k: v.to(device) for k,v in batch.items()}

                optimizer.zero_grad()

                enc_input = batch['input_ids']
                enc_mask = batch['attention_mask']
                targets = batch['labels']

                #targets: ['Hello', 'world', '<\s>']
                #Sentence: ['<s>', 'Hello', 'world']
                #Loss will ignore target '-100', alredy set
                #But when creating the dec input we must give the appropriate token to the
                #start of sentence token. The end of sentence is removed

                #Shift targets forward to get the decoder input
                dec_input = targets.clone().detach()
                dec_input = torch.roll(dec_input, shifts=1, dims=1)
                dec_input[:,0] = 65_001

                #also convert all -100 to pad token id
                dec_input = dec_input.masked_fill(
                    dec_input == -100, tokenizer.pad_token_id)

                #make the decoder input mask
                dec_mask = torch.ones_like(dec_input)
                dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)
                #Forward pass
                outputs = model(enc_input, dec_input, enc_mask, dec_mask)
                loss = criterion(outputs.transpose(2,1), targets)
                #Backward and optimize
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)

            model.eval()
            test_loss = []

            for batch in valid_loader:
                batch = {k: v.to(device) for k,v in batch.items()}

                enc_input = batch['input_ids']
                enc_mask = batch['attention_mask']
                targets = batch['labels']

                #shift the targets forwards to get the decoder input
                dec_input = targets.clone().detach()
                #Before we had the decoder inputs and shifted to the left (-1) to create the targets.
                #Now we have the targets and want to create the dec_inputs by shifting to right 1
                dec_input = torch.roll(dec_input, shifts=1, dims=1)
                dec_input[:, 0] = 65_001

                #change -100s to regular padding
                dec_input = dec_input.masked_fill(
                    dec_input == -100, tokenizer.pad_token_id)
                #make decoder input mask
                dec_mask = torch.ones_like(dec_input)
                dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id,0)

                outputs = model(enc_input, dec_input, enc_mask, dec_mask)
                loss = criterion(outputs.transpose(2,1), targets)
                test_loss.append(loss.item())

            test_loss = np.mean(test_loss)

            #Save losses
            train_losses[it] = train_loss
            test_losses[it] = test_loss
            dt = datetime.now()-t0

            print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
            Test Loss: {test_loss:.4f}, Duration: {dt}')
        return train_losses, test_losses