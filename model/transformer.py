import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util import clones

"""
self-Attention의 경우 Query Q, Key K, Value V를 입력으로 받아
MatMul(Q,K) -> Scale -> Masking(opt. Decoder) -> Softmax -> MatMul(result, V)

"""
class SelfAttention(nn.Module):
  def __init__(self):
    super(SelfAttention,self).__init__()
    self.matmul = torch.matmul()
    self.softmax = torch.softmax()

  def forward(self,query, key, value, mask=None):
    key_transpose = torch.transpose(key,-2,-1)          # (bath, -1, head_num, d_k)
    matmul_result = self.matmul(query,key_transpose)    # MatMul(Q,K)
    d_k = key.size()[-1]
    attention_score = matmul_result/math.sqrt(d_k)      # Scale

    if mask is not None:
      # 마스크가 있는경우리 뒤에 벡터들은 어텐션 받지 못하도록 마스킹 처리
      # 마스크가 0인 곳에 -1e9로 마스킹 처리
      attention_score = attention_score.masked_fill(mask ==0, -1e9)

    softmax_attention_score = torch.softmax(attention_score,dim=-1)                   # 어텐션 값
    result = self.matmul(softmax_attention_score,value)

    return result, softmax_attention_score


"""
멀티헤드 어텐션
MultiHead(Q,K,V) = Concat(head_1,head_2,...head_n)W^O
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
W^Q는 모델의 dimension x d_k
W^K는 모델의 dimension x d_k
W^V는 모델의 dimension x d_v
W^O는 d_v * head 갯수 x 모델 dimension
논문에서는 헤더의 갯수를 8개 사용
"""
class MultiHeadAttention(nn.Module):
  def __init__(self, head_num =8 , d_model = 512,dropout = 0.1):
    super(MaskedMultiHeadAttention,self).__init__()

    assert d_model % head_num == 0 # d_model % head_num == 0 이 아닌경우 에러메세지 발생

    self.head_num = head_num
    self.d_model = d_model
    self.d_k = self.d_v = d_model // head_num

    self.w_q = nn.Linear(d_model,d_model)
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)

    self.self_attention = SelfAttention()
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask = None):
    batche_num = query.size(0)

    query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

    attention_result, attention_score = self.self_attention(query, key, value, mask)

    # 원래의 모양으로 다시 변형해준다.
    # torch.continuos는 다음행과 열로 이동하기 위한 stride가 변형되어
    # 메모리 연속적으로 바꿔야 한다!
    # 참고 문서: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.h * self.d_k)

    return self.w_o(attention_result)

"""
Position-wise Feed-Forward Networks
FFN(x) = max(0,xW_1 + b_1)W_2+b2
입력과 출력은 모두 d_model의 dimension을 가지고
내부의 레이어는 d_model * 4의 dimension을 가진다.
"""
class FeedForward(nn.Module):
  def __init__(self,d_model, dropout = 0.1):
    super(FeedForward,self).__init__()
    self.w_1 = nn.Linear(d_model, d_model*4)
    self.w_2 = nn.Linear(d_model*4, d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    self.w_2(self.dropout(F.relu(self.w_1(x))))
"""
Layer Normalization
: layer의 hidden unit들에 대해서 mean과 variance를 구한다. 
nn.Parameter는 모듈 파라미터로 여겨지는 텐서
"""
class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm,self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
  def forward(self, x):
    mean = x.mean(-1, keepdim =True) # 평균
    std = x.std(-1, keepdim=True)    # 표준편차

    return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
  def __init__(self, size, dropout):
    super(ResidualConnection,self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout((sublayer(self.norm(x))))

class MaskedMultiHeadAttention(nn.Module):
  pass
"""
Encoder 블록은 FeedForward 레이어와 MultiHead 어텐션 레이어를 가진다.
"""
class Encoder(nn.Module):
  def __init__(self, d_model, head_num,dropout):
    super(Encoder,self).__init__()
    self.multi_head_attention = MultiHeadAttention(d_model= d_model)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)
    self.feed_forward = FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)

  def forward(self, input):
    x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x))
    x = self.residual_2(x, self.feed_forward)
    return x

"""
Decoder 블록은 FeedForward 레이어와 MultiHead 어텐션, Masked Multihead 어텐션 레이어를 가다.
"""

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    self.multi_head_attention = MultiHeadAttention()
    self.masked_multi_head_attention = MaskedMultiHeadAttention()
    self.feed_forward=FeedForward

  def forward(self, input, encoder_output):
    x = self.multi_head_attention(input)
    x = self.masked_multi_head_attention(x,encoder_output)
    x = self.feed_forward(x)

    return

class Embeddings(nn.Module):
  def __init__(self, vocab_num, d_model):
    super(Embeddings,self).__init__()
    self.emb = nn.Embedding(vocab_num,d_model)
    self.d_model = d_model
  def forward(self, x):
    """
    1) 임베딩 값에 math.sqrt(self.d_model)을 곱해주는 이유는 무엇인지 찾아볼것
    2) nn.Embedding에 다시 한번 찾아볼것
    """
    return self.emb(x) * math.sqrt(self.d_model)
"""
Positional Encoding
트랜스포머는 RNN이나 CNN을 사용하지 않기 때문에 입력에 순서 값을 반영해줘야 한다.
예) 나는 어제의 오늘
"""
class PositionalEncoding(nn.Module):
  def __init__(self, max_seq_len, d_model):
    super(PositionalEncoding,self).__init__()

class Transformer(nn.Module):
  def __init__(self):
    super(Transformer,self).__init__()
    self.encoder_bundle
