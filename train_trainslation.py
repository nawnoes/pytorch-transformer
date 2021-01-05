import torch
import torch.nn as nn

from torch.autograd import Variable
from model.util import subsequent_mask
from model.transformer import Transformer
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
    for i, batch in enumerate(dataset):
      out = model.forward(batch.src, batch.trg,
                          batch.src_mask, batch.trg_mask)
      loss = loss_compute(out, batch.trg_y, batch.ntokens)
      total_loss += loss
      total_tokens += batch.ntokens
      tokens += batch.ntokens
      if i % 50 == 1:
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
              (i, loss / batch.ntokens, tokens / elapsed))
        start = time.time()
        tokens = 0
    return total_loss / total_tokens

  def evaluate(self):
    pass

if __name__=='__main__':
  dataset =
  criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

  model = Transformer(vocab_num=22000,
                      d_model=512,
                      max_seq_len=512,
                      head_num=8,
                      dropout=0.1,
                      N=6)
  generator = model.generator
  model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                      torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

  trainer = TranslationTrainer()
  trainer.train(dataset=,
                model= model,
                loss_compute=SimpleLossCompute(generator,criterion,model_opt))