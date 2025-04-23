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
                batch = {k: v.to(device) for k,v in batch.items()}

                optimizer.zero_grad()

                enc_input = batch['input_ids']
                enc_mask = batch['attention_mask']
                targets = batch['labels']

                dec_input = targets.clone().detach()
                dec_input = torch.roll(dec_input, shifts=1, dims=1)
                dec_input[:,0] = 65_001

                dec_input = dec_input.masked_fill(
                    dec_input == -100, tokenizer.pad_token_id)

                dec_mask = torch.ones_like(dec_input)
                dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id, 0)

                outputs = model(enc_input, dec_input, enc_mask, dec_mask)
                loss = criterion(outputs.transpose(2,1), targets)
                
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

                dec_input = targets.clone().detach()

                dec_input = torch.roll(dec_input, shifts=1, dims=1)
                dec_input[:, 0] = 65_001

                dec_input = dec_input.masked_fill(
                    dec_input == -100, tokenizer.pad_token_id)
                
                dec_mask = torch.ones_like(dec_input)
                dec_mask = dec_mask.masked_fill(dec_input == tokenizer.pad_token_id,0)

                outputs = model(enc_input, dec_input, enc_mask, dec_mask)
                loss = criterion(outputs.transpose(2,1), targets)
                test_loss.append(loss.item())

            test_loss = np.mean(test_loss)

            train_losses[it] = train_loss
            test_losses[it] = test_loss
            dt = datetime.now()-t0

            print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
            Test Loss: {test_loss:.4f}, Duration: {dt}')
        return train_losses, test_losses