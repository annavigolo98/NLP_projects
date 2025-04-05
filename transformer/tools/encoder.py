import torch.nn as nn
from transformer.tools.multi_head_attention import MultiHeadAttention
from transformer.tools.positional_encoding import PositionalEncoding




class EncoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob)
            )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, pad_mask=None):
        x = self.ln1(x+self.mha(x,x,x,pad_mask))
        x = self.ln2(x+self.ann(x))
        x = self.dropout(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               dropout_prob):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            EncoderBlock(
                d_k,
                d_model,
                n_heads,
                max_len,
                dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, pad_mask)
        #MANY TO MANY

        x = self.ln(x)
        return x
    