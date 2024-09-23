import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        self.embedding = nn.Embedding(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        # 미리 max_len 크기로 positional encoding을 계산하여 저장
        position = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # exponential term

        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스에 대해 사인 함수
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스에 대해 코사인 함수
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.embedding(x) + self.pe[:, :x.size(1), :]