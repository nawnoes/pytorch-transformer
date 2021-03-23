import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from model.util import clones
from transformers.activations import get_activation

"""
self-Attention의 경우 Query Q, Key K, Value V를 입력으로 받아
MatMul(Q,K) -> Scale -> Masking(opt. Decoder) -> Softmax -> MatMul(result, V)

"""

def self_attention(query, key, value, mask=None):
  key_transpose = torch.transpose(key,-2,-1)                      # (bath, head_num, d_k, token_)
  matmul_result = torch.matmul(query,key_transpose)                # MatMul(Q,K)
  d_k = query.size()[-1]
  attention_score = matmul_result/math.sqrt(d_k)                  # Scale

  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -1e20)

  softmax_attention_score = F.softmax(attention_score,dim=-1)  # 어텐션 값
  result = torch.matmul(softmax_attention_score,value)

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
    super(MultiHeadAttention,self).__init__()

    # print(d_model % head_num)
    # assert d_model % head_num != 0 # d_model % head_num == 0 이 아닌경우 에러메세지 발생

    self.head_num = head_num
    self.d_model = d_model
    self.d_k = self.d_v = d_model // head_num

    self.w_q = nn.Linear(d_model,d_model)
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)

    self.self_attention = self_attention
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask = None):
    if mask is not None:
      # Same mask applied to all h heads.
      mask = mask.unsqueeze(1)

    batche_num = query.size(0)

    query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

    attention_result, attention_score = self.self_attention(query, key, value, mask)

    # 원래의 모양으로 다시 변형해준다.
    # torch.continuos는 다음행과 열로 이동하기 위한 stride가 변형되어
    # 메모리 연속적으로 바꿔야 한다!
    # 참고 문서: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.head_num * self.d_k)


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
    return self.w_2(self.dropout(F.relu(self.w_1(x))))
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

"""
Encoder 블록은 FeedForward 레이어와 MultiHead 어텐션 레이어를 가진다.
"""
class Encoder(nn.Module):
  def __init__(self, d_model, head_num, dropout):
    super(Encoder,self).__init__()
    self.multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward = FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)

  def forward(self, input, mask):
    x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
    x = self.residual_2(x, lambda x: self.feed_forward(x))
    return x

"""
Decoder 블록은 FeedForward 레이어와 MultiHead 어텐션, Masked Multihead 어텐션 레이어를 가진다.
MaskedMultiHeadAttention -> MultiHeadAttention(encoder-decoder attention) -> FeedForward
"""

class Decoder(nn.Module):
  def __init__(self, d_model,head_num, dropout):
    super(Decoder,self).__init__()
    self.masked_multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.encoder_decoder_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward= FeedForward(d_model)
    self.residual_3 = ResidualConnection(d_model,dropout=dropout)


  def forward(self, target, encoder_output, target_mask, encoder_mask):
    # target, x, target_mask, input_mask
    x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask))
    x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encoder_mask))
    x = self.residual_3(x, self.feed_forward)

    return x

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
PE (pos,2i) = sin(pos/10000^(2i/d_model))
PE (pos,2i+1) = cos(pos/10000^(2i/d_model)) 
"""
class PositionalEncoding(nn.Module):
  def __init__(self, max_seq_len, d_model,dropout=0.1):
    super(PositionalEncoding,self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_seq_len, d_model)

    position = torch.arange(0,max_seq_len).unsqueeze(1)
    base = torch.ones(d_model//2).fill_(10000)
    pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model,dtype=torch.float32)
    div_term = torch.pow(base,pow_term)

    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)

    pe = pe.unsqueeze(0)

    # pe를 학습되지 않는 변수로 등록
    self.register_buffer('positional_encoding', pe)

  def forward(self, x):
    x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)


class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)

class Generator(nn.Module):
  def __init__(self, d_model, vocab_num):
    super(Generator, self).__init__()
    self.proj_1 = nn.Linear(d_model, d_model*4)
    self.proj_2 = nn.Linear(d_model*4, vocab_num)

  def forward(self, x):
    x = self.proj_1(x)
    x = self.proj_2(x)
    return x
class Transformer(nn.Module):
  def __init__(self,vocab_num, d_model, max_seq_len, head_num, dropout, N):
    super(Transformer,self).__init__()
    self.embedding = Embeddings(vocab_num, d_model)
    self.positional_encoding = PositionalEncoding(max_seq_len,d_model)

    self.encoders = clones(Encoder(d_model=d_model, head_num=head_num, dropout=dropout), N)
    self.decoders = clones(Decoder(d_model=d_model, head_num=head_num, dropout=dropout), N)

    self.generator = Generator(d_model, vocab_num)

  def forward(self, input, target, input_mask, target_mask, labels=None):
      x = self.positional_encoding(self.embedding(input))
      for encoder in self.encoders:
        x = encoder(x, input_mask)

      target = self.positional_encoding(self.embedding(target))
      for decoder in self.decoders:
        # target, encoder_output, target_mask, encoder_mask)
        target = decoder(target, x, target_mask, input_mask)

      lm_logits = self.generator(target)
      loss = None
      if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

      return lm_logits, loss
  def encode(self,input, input_mask):
    x = self.positional_encoding(self.embedding(input))
    for encoder in self.encoders:
      x = encoder(x, input_mask)
    return x

  def decode(self, encode_output, encoder_mask, target, target_mask):
    target = self.positional_encoding(self.embedding(target))
    for decoder in self.decoders:
      #target, encoder_output, target_mask, encoder_mask
      target = decoder(target, encode_output, target_mask, encoder_mask)

    lm_logits = self.generator(target)

    return lm_logits
class TransformerMRCHead(nn.Module):
  def __init__(self, dim, num_labels,hidden_dropout_prob=0.3):
    super().__init__()
    self.dense = nn.Linear(dim, 1*dim)
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.out_proj = nn.Linear(1*dim,num_labels)

  def forward(self, x, **kwargs):
    # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class TransformerMRCModel(nn.Module):
  def __init__(self, vocab_size, dim, depth, max_seq_len, head_num, num_labels=2, causal=False, dropout_prob=0.2):
    super().__init__()
    self.transformer = TransformerLM(
      vocab_size=vocab_size,
      dim=dim,
      depth=depth,
      max_seq_len=max_seq_len,
      head_num=head_num,
    )
    self.mrc_head = TransformerMRCHead(dim, num_labels)

  def forward(self,
              input_ids=None,
              input_mask=None,
              start_positions=None,
              end_positions=None,
              **kwargs):
    # 1. transformer의 출력
    _, outputs = self.transformer(input_ids, input_mask)

    # 2. mrc를 위한
    logits = self.mrc_head(outputs)

    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    if start_positions is not None and end_positions is not None:
      # If we are on multi-GPU, split add a dimension
      if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
      # sometimes the start/end positions are outside our model inputs, we ignore these terms
      ignored_index = start_logits.size(1)
      start_positions.clamp_(0, ignored_index)
      end_positions.clamp_(0, ignored_index)

      loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
      start_loss = loss_fct(start_logits, start_positions)
      end_loss = loss_fct(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2
      return total_loss
    else:
      return start_logits, end_logits

class TransformerLM(nn.Module):
  def __init__(self, vocab_size, dim=512,  depth= 12, max_seq_len=512, head_num=8, dropout= 0.1):
    super(TransformerLM,self).__init__()

    self.token_emb= nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim,max_seq_len)
    self.encoders = clones(Encoder(d_model=dim, head_num=head_num, dropout=dropout), depth)
    self.norm = nn.LayerNorm(dim)
    self.lm_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, vocab_size)
            )

  def forward(self, input_ids, input_mask):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x)

    for encoder in self.encoders:
      x = encoder(x, input_mask)
    x = self.norm(x)

    return self.lm_head(x), x  # lm_head, performer_embedding



if __name__=="__main__":
  pass
