import torch.nn as nn
from transformer.tools.multi_head_attention import MultiHeadAttention
from transformer.tools.positional_encoding import PositionalEncoding




class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_k, 
                 d_m, 
                 heads, 
                 max_causal_mask_length, 
                 dropout_probability=0.1):
        
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_m)
        self.layer_norm_2 = nn.LayerNorm(d_m)
        self.multi_head_attention = MultiHeadAttention(d_k, 
                                                       d_m, 
                                                       heads, 
                                                       max_causal_mask_length, 
                                                       causal=False)
        
        self.ann = nn.Sequential(
            nn.Linear(d_m, d_m*4),
            nn.GELU(),
            nn.Linear(d_m*4, d_m),
            nn.Dropout(dropout_probability)
            )

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x, pad_mask=None):
        x = self.layer_norm_1(x+self.multi_head_attention(x,x,x,pad_mask))
        x = self.layer_norm_2(x+self.ann(x))
        x = self.dropout(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self,
                d_k,
                d_m,
                heads,
                vocaboulary_size,
                max_length,
                n_encoder_blocks,
                dropout_probability):

        super().__init__()

        self.embedding = nn.Embedding(vocaboulary_size, d_m)
        self.positional_encoding = PositionalEncoding(d_m, max_length, dropout_probability)
        transformer_blocks = [
            EncoderBlock(
                d_k,
                d_m,
                heads,
                max_length,
                dropout_probability) for _ in range(n_encoder_blocks)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.layer_norm = nn.LayerNorm(d_m)

    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, pad_mask)

        x = self.layer_norm(x)
        return x
    