"""
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


class NoamOpt:
  "Optim wrapper that implements rate."

  def __init__(self, model_size, factor, warmup, optimizer):
    self.optimizer = optimizer
    self._step = 0
    self.warmup = warmup
    self.factor = factor
    self.model_size = model_size
    self._rate = 0

  def step(self):
    "Update parameters and rate"
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()

  def rate(self, step=None):
    "Implement `lrate` above"
    if step is None:
      step = self._step
    return self.factor * \
           (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
  return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                 torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def visualize_NoamOpt():
  # Three settings of the lrate hyperparameters.
  opts = [NoamOpt(512, 1, 4000, None),
          NoamOpt(512, 1, 8000, None),
          NoamOpt(256, 1, 4000, None)]
  plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
  plt.legend(["512:4000", "512:8000", "256:4000"])
  plt.show

class LabelSmoothing(nn.Module):
  def __init__(self, size, padding_idx, smoothing=0.0):
    super(LabelSmoothing, self).__init__()
    self.criterion = nn.KLDivLoss(size_average=False)
    self.padding_idx = padding_idx
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size
    self.true_dist = None

  def forward(self, x, target):
    assert x.size(1) == self.size
    true_dist = x.data.clone()
    true_dist.fill_(self.smoothing / (self.size - 2))
    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
    true_dist[:, self.padding_idx] = 0
    mask = torch.nonzero(target.data == self.padding_idx)
    if mask.dim() > 0:
      true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(x, Variable(true_dist, requires_grad=False))

def visualize_label_smoothing():
  # Example of label smoothing.
  crit = LabelSmoothing(5, 0, 0.4)
  predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                               [0, 0.2, 0.7, 0.1, 0],
                               [0, 0.2, 0.7, 0.1, 0]])
  v = crit(Variable(predict.log()),
           Variable(torch.LongTensor([2, 1, 0])))

  # Show the target distributions expected by the system.
  plt.imshow(crit.true_dist)
  crit = LabelSmoothing(5, 0, 0.1)
  def loss(x):
      d = x + 3 * 1
      predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                   ])
      #print(predict)
      return crit(Variable(predict.log()),
                   Variable(torch.LongTensor([1]))).data[0]
  plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
  plt.show()

class SimpleLossCompute:
  "A simple loss compute and train function."
  def __init__(self, generator, criterion, opt=None):
    self.generator = generator
    self.criterion = criterion
    self.opt = opt

  def __call__(self, x, y, norm):
    x = self.generator(x)
    loss = self.criterion(x[:,1:].contiguous().view(-1, x.size(-1)), y[:,:-1].contiguous().view(-1)) / norm.sum()

    loss.backward()
    if self.opt is not None:
      self.opt.step()
      self.opt.optimizer.zero_grad()
    return loss.data * norm.sum()

def load_csv(file_path):
  print(f'Load Data | file path: {file_path}')
  with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    lines = []
    for line in csv_reader:
      line[0] = line[0].replace(';','')
      lines.append(line)
  print(f'Load Complete | file path: {file_path}')

  return lines

if __name__=="__main__":
  path = './data/ko-en-translation.csv'
  lines = load_csv(path)
