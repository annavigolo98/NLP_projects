import os
import torch.nn as nn
import json
import torch

from transformer.tools.decoder import Decoder
from transformer.tools.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
        return dec_output

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

        config_dict = {
            'encoder_config': self.config['encoder_config'],
            'decoder_config': self.config['decoder_config']
        }

        with open(f"{save_directory}/config.json", 'w') as f:
            json.dump(config_dict, f)

    
    
    def generate(self, 
                  encoder_input, #batch_size x seq_len
                  encoder_mask, #batch_size x seq_len
                  max_tokens, 
                  device):
        
        bos_token=65_001
        eos_token = 0
            
        encoder_output = self.encoder(encoder_input, encoder_mask)
        batch_size = encoder_input.size(0)

        decoder_input_ids = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        

        for _ in range(max_tokens):
            decoder_attn_mask = torch.ones_like(decoder_input_ids, device=device)
            dec_output = self.decoder(
                encoder_output,
                decoder_input_ids,
                encoder_mask,
                decoder_attn_mask
            )

            next_token = torch.argmax(dec_output[:, -1, :], dim=-1)
            next_token = next_token.masked_fill(is_finished, eos_token) #If sentence is finished in the row, predicts only pad

            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(1)], dim=1) #add a dim to next token to append to decoder_input_ids of shape [batch_size x seq_len]

            is_finished |= next_token == eos_token

            if is_finished.all():
                break

        
        translation_batch = decoder_input_ids[:, 1:]
        return translation_batch


    @classmethod
    def from_pretrained(cls, save_directory):
        with open(f"{save_directory}/config.json", 'r') as f:
            config_dict = json.load(f)
        
        encoder = Encoder(**config_dict['encoder_config'])
        decoder = Decoder(**config_dict['decoder_config'])
        
        model = cls(encoder, decoder, config_dict)
        
        model.load_state_dict(torch.load(f"{save_directory}/pytorch_model.bin"))
        
        return model, encoder, decoder