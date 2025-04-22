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


    @classmethod
    def from_pretrained(cls, save_directory):
        with open(f"{save_directory}/config.json", 'r') as f:
            config_dict = json.load(f)
        
        encoder = Encoder(**config_dict['encoder_config'])
        decoder = Decoder(**config_dict['decoder_config'])
        
        model = cls(encoder, decoder, config_dict)
        
        model.load_state_dict(torch.load(f"{save_directory}/pytorch_model.bin"))
        
        return model, encoder, decoder