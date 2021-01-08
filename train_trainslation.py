import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from model.transformer import Transformer
from transformers import BertTokenizer
from dataset import TranslationDataset
from torch.utils.data import DataLoader
from util import LabelSmoothing, NoamOpt, SimpleLossCompute
import time

class TranslationTrainer():
  def __init__(self):
    pass

  def train(self, dataset, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, data in enumerate(dataset):
      input = data[0]
      target = data[2][:,:-1]
      input_mask = data[1]
      target_mask = data[3]
      token_num = data[4]

      out = model.forward(input, target, input_mask, target_mask)

      loss = loss_compute(out, target[...,1:],token_num)
      total_loss += loss
      total_tokens += data['token_num']
      tokens +=data['token_num']
      if i % 50 == 1:
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
              (i, loss / data['token_num'], tokens / elapsed))
        start = time.time()
        tokens = 0
    return total_loss / total_tokens


if __name__=='__main__':
  vocab_path = './data/wiki-vocab.txt'
  data_path = './data/test.csv'
  # model setting
  vocab_num = 22000
  max_length = 512
  d_model = 512
  head_num = 8
  dropout = 0.1
  N = 3

  tokenizer = BertTokenizer(vocab_path)

  # hyper parameter
  batch_size = 8
  padding_idx =tokenizer.pad_token_id

  dataset = TranslationDataset(tokenizer=tokenizer,file_path=data_path,max_length = max_length)
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  criterion = LabelSmoothing(size=batch_size, padding_idx=tokenizer.pad_token_id, smoothing=0.0)

  model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N)

  generator = model.generator
  model_opt = NoamOpt(model.embedding.d_model, 1, 400,
                      torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

  trainer = TranslationTrainer()
  trainer.train(dataset= train_loader,
                model= model,
                loss_compute=SimpleLossCompute(generator,criterion,model_opt))