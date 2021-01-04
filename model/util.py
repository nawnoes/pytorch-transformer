from torch.nn import ModuleList
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model.transformer import PositionalEncoding
"""
ModuleList는 목록에 하위 모듈을 보관하것
이때 모듈들은 파이썬 리스트들 처럼 인덱스를 사용할 수 있다.
"""
def clones(module, N):
  return ModuleList([copy.deepcopy(module) for i in range(N)])

"""
디코더에서 어텐션 스코어 매트릭스에서
이후의 값들에 대해 -∞으로 마스킹 처리해주기 위한 함수
(1, size, size)의 마스크를 리턴한다.
"""
def subsequent_mask(size):
  "Mask out subsequent positions."
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(subsequent_mask) == 0

def visualize_subsequent_mask():
  plt.figure(figsize=(5, 5))
  plt.imshow(subsequent_mask(20)[0])

def visualize_positional_encoding():
  plt.figure(figsize=(15, 5))
  pe = PositionalEncoding(5000, 20, 0)
  x = Variable(torch.zeros(1, 100, 20))

  y = pe.forward(x)
  plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
  plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
  plt.show()