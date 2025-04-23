import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, causal=False):
        super().__init__()
        self.d_k=d_k
        self.n_heads=n_heads

        self.key = nn.Linear(d_model, d_k*n_heads)
        self.query = nn.Linear(d_model, d_k*n_heads)
        self.value = nn.Linear(d_model, d_k*n_heads)

        self.fc = nn.Linear(d_k*n_heads, d_model)

        self.causal = causal
        if causal:
            cm = torch.tril(torch.ones(max_len, max_len))
            self.register_buffer(
            "causal_mask",
            cm.view(1, 1, max_len, max_len)
            )

    def forward(self, q, k, v, pad_mask=None):
        q = self.query(q) # N x T_out x h*d_k
        k = self.key(k) # N x T_in x h*d_k
        v = self.value(v) # N x T_in x h*d_k

        N = q.shape[0]
        T_output = q.shape[1]
        T_input = k.shape[1]

        #(N, T, h, d_k)->(N, h, T, d_k)
        q = q.view(N, T_output, self.n_heads, self.d_k).transpose(1,2)
        k = k.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)

        #(N, h, T_out, d_k) x (N, h, d_k, T_in) -> (N, h, T_out, T_in)
        attn_scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(
            pad_mask[:, None, None, :] == 0, float('-inf'))
        if self.causal:
            attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T_output, :T_input] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        #(N, h, T_out, T_in)x(N, h, T_in, d_k)->(N, h, T_out, d_k)
        A = attn_weights @ v
        A = A.transpose(1,2) #(N, T_out, h, d_k)
        A = A.contiguous().view(N, T_output, self.d_k * self.n_heads) # (N, T_out, h*d)
        return self.fc(A)