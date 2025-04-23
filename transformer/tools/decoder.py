import torch.nn as nn
from transformer.tools.multi_head_attention import MultiHeadAttention
from transformer.tools.positional_encoding import PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=True)
        self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob))

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        x = self.ln1(dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask))
        x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))

        x = self.ln3(x + self.ann(x))
        x = self.dropout(x)
        return x



class Decoder(nn.Module):
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
            DecoderBlock(
                d_k,
                d_model,
                n_heads,
                max_len,
                dropout_prob) for _ in range(n_layers)]

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        x = self.embedding(dec_input)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(enc_output, x, enc_mask, dec_mask)
        x = self.ln(x)
        x = self.fc(x)
        return x