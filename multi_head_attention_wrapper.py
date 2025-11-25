import torch
import torch.nn as nn

from causal_attention import CausalAttention

class MultiHeadAttentionWrapper(nn.Module):

  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False) -> None:
    super().__init__()
    # self.heads = nn.ModuleList(
    #   [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
    #   for _ in range(num_heads)]
    # )
    assert(d_out % num_heads == 0), \
      "d_out must be divisible by num_heads"
    
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_in, d_out) # Linear layer to combine head outputs
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
  
  def forward(self, x):
    b, num_tokens, d_in = x.shape
    # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`
    # this will result in errors in the mask creation further below.
    # In practice, this is not a problem since the LLM (chapters 4 to 7) ensures that inputs
    # do not exceed `context_length` before reaching this forward method.

    keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
    queries = self.W_query(x)
    values = self.W_value(x)

    # We implicitly split the matrix by adding a `num_heads` dimension
    # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    # Transpose: 
