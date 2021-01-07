from torch.nn import ModuleList
import copy
import torch
import numpy as np
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
