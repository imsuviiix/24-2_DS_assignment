import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO
        #query, key, value: (n_batch, n_heads, n_seq, d_k)
        d_k = k.shape[-1] 
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # padding mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스를 적용하여 attention weights 계산
        attention_weights = F.softmax(attn_scores, dim=-1) # (n_batch, n_heads, n_seq, n_seq)
        
        # attention weights @ V
        output = torch.matmul(attention_weights, v) # (n_batch, n_heads, n_seq, d_k)
        
        return output, attention_weights
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        # Q, K, V: (n_batch, n_seq, d_model)
        # mask: (n_batch, n_seq, n_seq)
        # return: (n_batch, h, n_seq, d_k)
        batch_size = Q.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        
        # Query, Key, Value에 대해 각각 projection
        Q = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model//self.n_heads).transpose(1, 2)
        K = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model//self.n_heads).transpose(1, 2)
        V = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model//self.n_heads).transpose(1, 2)
        
        # Scaled Dot-Product Attention을 각 head에 대해 적용
        attention_output, attention_weights= self.attention(Q, K, V, mask)
        
        # 모든 head의 출력을 결합
        attention_output = attention_output.transpose(1, 2)
        attention_output= attention_output.contiguous().view(batch_size, -1, self.d_model*self.n_heads) # (n_batch, n_seq, d_model)
        
        # 최종 선형 계층 적용
        output = self.fc(attention_output)
        
        return output
        