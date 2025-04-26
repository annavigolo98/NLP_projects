from datetime import datetime
import numpy as np
from pydantic import BaseModel
import torch
from tqdm import tqdm



class Train(BaseModel):
    def train(self,
             model,
             loss_function,
             optimizer, 
             train_dataloader, 
             valid_dataloader, 
             n_epochs, 
             device,
             tokenizer):
        
        train_losses = np.zeros(n_epochs)
        test_losses = np.zeros(n_epochs)

        for it in range(n_epochs):
            model.train()
            t0 = datetime.now()
            train_loss = []
            for batch in tqdm(train_dataloader):
                batch = {k: v.to(device) for k,v in batch.items()}

                optimizer.zero_grad()

                encoder_input = batch['input_ids']
                encoder_mask = batch['attention_mask']
                targets = batch['labels']

                decoder_input = targets.clone().detach()
                decoder_input = torch.roll(decoder_input, shifts=1, dims=1)
                decoder_input[:,0] = 65_001

                decoder_input = decoder_input.masked_fill(
                    decoder_input == -100, tokenizer.pad_token_id)

                decoder_mask = torch.ones_like(decoder_input)
                decoder_mask = decoder_mask.masked_fill(decoder_input == tokenizer.pad_token_id, 0)

                outputs = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_function(outputs.transpose(2,1), targets)
                
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)

            model.eval()
            test_loss = []

            for batch in valid_dataloader:
                batch = {k: v.to(device) for k,v in batch.items()}

                encoder_input = batch['input_ids']
                encoder_mask = batch['attention_mask']
                targets = batch['labels']

                decoder_input = targets.clone().detach()

                decoder_input = torch.roll(decoder_input, shifts=1, dims=1)
                decoder_input[:, 0] = 65_001

                decoder_input = decoder_input.masked_fill(
                    decoder_input == -100, tokenizer.pad_token_id)
                
                decoder_mask = torch.ones_like(decoder_input)
                decoder_mask = decoder_mask.masked_fill(decoder_input == tokenizer.pad_token_id,0)

                outputs = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_function(outputs.transpose(2,1), targets)
                test_loss.append(loss.item())

            test_loss = np.mean(test_loss)

            train_losses[it] = train_loss
            test_losses[it] = test_loss
            dt = datetime.now()-t0

            print(f'Epoch {it+1}/{n_epochs}, Train Loss: {train_loss:.4f}, \
            Test Loss: {test_loss:.4f}, Duration: {dt}')
        return train_losses, test_losses