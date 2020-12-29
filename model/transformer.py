import math
import torch
import torch.nn as nn

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
    key_transpose = torch.transpose(key,1,2)
    x = self.matmul(query,key_transpose)          # MatMul(Q,K)
    d_k = key.size()[-1]
    x = x/math.sqrt(d_k)                          # Scale
    if mask is not None:
      # 마스크가 있는경우리 뒤에 벡터들은 어텐션 받지 못하도록 마스킹 처리
      pass
    x = torch.softmax(x,dim=-1)                   # 어텐션 값
    x = self.matmul(x,value)

    return x


"""
멀티헤드 어텐션
MultiHead(Q,K,V) = Concat(head_1,head_2,...head_n)W^O
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
W^Q는 모델의 dimension x d_k
W^K는 모델의 dimension x d_k
W^V는 모델의 dimension x d_v
W^O는 d_v * head 갯수 x 모델 dimension
논문에서는 헤더의 갯수를 8개 사
"""
class MultiHeadAttention(nn.Module):
  def __init__(self, head_num, d_model):
    super(MaskedMultiHeadAttention,self).__init__()
    self.head_num =head_num
    self.d_model = d_model


  def forward(self, input ):
class FeedForward(nn.Module):
  pass
class MaskedMultiHeadAttention(nn.Module):
  pass
"""
Encoder 블록은 FeedForward 레이어와 MultiHead 어텐션 레이어를 가진다.
"""
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder,self).__init__()
    self.multi_head_attention = MultiHeadAttention()
    self.feed_forward = FeedForward()

  def forward(self, input):
    x = self.multi_head_attention(input)
    x = self.feed_forward(x)

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


class Transformer(nn.Module):
  pass
