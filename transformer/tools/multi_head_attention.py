import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_k, 
                 d_m, 
                 heads, 
                 max_causal_mask_length, 
                 causal=False):
        
        super().__init__()

        self.heads=heads
        self.d_k=d_k
        

        self.W_q = nn.Linear(d_m, d_k*heads)
        self.W_k = nn.Linear(d_m, d_k*heads)
        self.W_v = nn.Linear(d_m, d_k*heads)

        self.linear_layer = nn.Linear(d_k*heads, d_m)

        self.causal = causal
        if causal:
            cm = torch.tril(torch.ones(max_causal_mask_length, max_causal_mask_length))
            self.register_buffer(
            "causal_mask",
            cm.view(1, 1, max_causal_mask_length, max_causal_mask_length)
            )

    def forward(self, q, k, v, pad_mask=None):
        q = self.W_q(q) # N x T_out x h*d_k
        k = self.W_k(k) # N x T_in x h*d_k
        v = self.W_v(v) # N x T_in x h*d_k

        N = q.shape[0]
        T_out = q.shape[1]
        T_in = k.shape[1]

        #(N, T, h, d_k)->(N, h, T, d_k)
        q = q.view(N, T_out, self.heads, self.d_k).transpose(1,2)
        k = k.view(N, T_in, self.heads, self.d_k).transpose(1,2)
        v = v.view(N, T_in, self.heads, self.d_k).transpose(1,2)

        #(N, h, T_out, d_k) x (N, h, d_k, T_in) -> (N, h, T_out, T_in)
        attention_scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        if pad_mask is not None:
            attention_scores = attention_scores.masked_fill(
            pad_mask[:, None, None, :] == 0, float('-inf'))
        if self.causal:
            attention_scores = attention_scores.masked_fill(
            self.causal_mask[:, :, :T_out, :T_in] == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)

        #(N, h, T_out, T_in)x(N, h, T_in, d_k)->(N, h, T_out, d_k)
        A = attention_weights @ v
        A = A.transpose(1,2) #(N, T_out, h, d_k)
        A = A.contiguous().view(N, T_out, self.d_k * self.heads) # (N, T_out, h*d_k)
        return self.linear_layer(A)