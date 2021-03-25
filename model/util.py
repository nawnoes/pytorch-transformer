from functools import reduce

from torch.nn import ModuleList
import torch.nn.functional as F
import copy
import math
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

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def temperature_sampling(logits, temperature):
  if temperature is None or temperature == 0.0:
    return torch.argmax(logits)
  probs = F.softmax(logits / temperature)
  pred_ids = probs.cpu().multinomial(probs.size()[1], replacement=False)
  return pred_ids