import torch.nn as nn
from transformer.tools.multi_head_attention import MultiHeadAttention
from transformer.tools.positional_encoding import PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(self,
                d_k,
                d_m, 
                heads, 
                max_causal_mask_length, 
                dropout_probability=0.1):
        
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_m)
        self.layer_norm_2 = nn.LayerNorm(d_m)
        self.layer_norm_3 = nn.LayerNorm(d_m)

        self.multi_head_attention_casual = MultiHeadAttention(d_k,
                                                            d_m,
                                                            heads, 
                                                            max_causal_mask_length, 
                                                            causal=True)
        
        self.multi_head_attention = MultiHeadAttention(d_k, 
                                                       d_m, 
                                                       heads, 
                                                       max_causal_mask_length, 
                                                       causal=False)
        
        self.ann = nn.Sequential(
            nn.Linear(d_m, d_m*4),
            nn.GELU(),
            nn.Linear(d_m*4, d_m),
            nn.Dropout(dropout_probability))

        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, enc_output, decoder_input, encoder_mask=None, decoder_mask=None):

        x = self.layer_norm_1(decoder_input + self.multi_head_attention_casual(decoder_input, decoder_input, decoder_input, decoder_mask))
        x = self.layer_norm_2(x + self.multi_head_attention(x, enc_output, enc_output, encoder_mask))
        x = self.layer_norm_3(x + self.ann(x))
        x = self.dropout(x)

        return x



class Decoder(nn.Module):
    def __init__(self,
               d_k,
               d_m,
               heads,
               vocaboulary_size,
               max_length,
               n_decoder_blocks,
               dropout_probability):
        
        super().__init__()

        self.embedding = nn.Embedding(vocaboulary_size, d_m)
        self.positional_encoding = PositionalEncoding(d_m, max_length, dropout_probability)
        transformer_blocks = [
            DecoderBlock(
                d_k,
                d_m,
                heads,
                max_length,
                dropout_probability) for _ in range(n_decoder_blocks)]

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.layer_norm = nn.LayerNorm(d_m)
        self.linear_layer = nn.Linear(d_m, vocaboulary_size)

    def forward(self, encoder_output, decoder_input, encoder_mask=None, decoder_mask=None):
        x = self.embedding(decoder_input)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(encoder_output, x, encoder_mask, decoder_mask)
        x = self.layer_norm(x)
        x = self.linear_layer(x)
        return x