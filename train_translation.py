import warnings
warnings.filterwarnings("ignore")
import sys

sys.path.append('/content/drive/My Drive/Colab Notebooks/transformer')

import torch
from model.transformer import Transformer
from transformers import BertTokenizer
from dataset import TranslationDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math

import time


class TranslationTrainer():
  def __init__(self,
               dataset,
               tokenizer,
               model,
               max_len,
               device,
               model_name,
               checkpoint_path,
               batch_size,
               ):
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.model = model
    self.max_len = max_len
    self.model_name = model_name
    self.checkpoint_path = checkpoint_path
    self.device = device
    self.ntoken = tokenizer.vocab_size
    self.batch_size = batch_size

  def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
    dataset_len = len(self.dataset)
    eval_len = int(dataset_len * train_test_split)
    train_len = dataset_len - eval_len
    train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=train_shuffle, )
    eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=eval_shuffle)

    return train_loader, eval_loader

  def train(self, epochs, train_dataset, eval_dataset, optimizer, scheduler):
    self.model.train()
    total_loss = 0.
    global_steps = 0
    start_time = time.time()
    losses = {}
    best_val_loss = float("inf")
    best_model = None

    self.model.to(self.device)
    for epoch in range(epochs):
      epoch_start_time = time.time()

      pb = tqdm(enumerate(train_dataset),
                desc=f'Epoch-{epoch} Iterator',
                total=len(train_dataset),
                bar_format='{l_bar}{bar:10}{r_bar}'
                )
      for i, data in pb:
        input = data[0].to(self.device)
        target = data[2].to(self.device)
        input_mask = data[1].to(self.device)
        target_mask = data[3].to(self.device)

        optimizer.zero_grad()
        generator_logit, loss = self.model.forward(input, target, input_mask, target_mask, labels=target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        losses[global_steps] = loss.item()
        total_loss += loss.item()
        log_interval = 1
        global_steps += 1

        if i % log_interval == 0 and i > 0:
          cur_loss = total_loss / log_interval
          elapsed = time.time() - start_time
          # print('| epoch {:3d} | {:5d}/{:5d} batches | '
          #       'lr {:02.2f} | ms/batch {:5.2f} | '
          #       'loss {:5.2f} | ppl {:8.2f}'.format(
          #         epoch, i, len(train_dataset), scheduler.get_lr()[0],
          #         elapsed * 1000 / log_interval,
          #         cur_loss, math.exp(cur_loss)))
          pb.set_postfix_str('| epoch {:3d} | {:5d}/{:5d} batches | '
                             'lr {:02.2f} | ms/batch {:5.2f} | '
                             'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, i, len(train_dataset), scheduler.get_lr()[0],
            elapsed * 1000 / log_interval,
            cur_loss, math.exp(cur_loss)))
          total_loss = 0
          start_time = time.time()
          self.save(epoch, self.model, optimizer, losses, global_steps)
      val_loss = self.evaluate(eval_dataset)
      self.model.train()
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
      print('-' * 89)
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

      scheduler.step()

  def evaluate(self, dataset):
    self.model.eval()  # 평가 모드를 시작합니다.
    total_loss = 0.

    self.model.to(self.device)
    with torch.no_grad():
      for i, data in enumerate(dataset):
        input = data[0].to(self.device)
        target = data[2].to(self.device)
        input_mask = data[1].to(self.device)
        target_mask = data[3].to(self.device)

        generator_logit, loss = self.model.forward(input, target, input_mask, target_mask, labels=target)
        total_loss += loss.item()

    return total_loss / (len(dataset) - 1)

  def save(self, epoch, model, optimizer, losses, train_step):
    torch.save({
      'epoch': epoch,  # 현재 학습 epoch
      'model_state_dict': model.state_dict(),  # 모델 저장
      'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
      'losses': losses,  # Loss 저장
      'train_step': train_step,  # 현재 진행한 학습
    }, f'{self.checkpoint_path}/{self.model_name}.pth')


if __name__ == '__main__':
  #dir_path = '/content/drive/My Drive/Colab Notebooks/transformer'
  dir_path = '.'
  vocab_path = f'{dir_path}/data/wiki-vocab.txt'
  data_path = f'{dir_path}/data/ko-en-translation.csv'
  checkpoint_path = f'{dir_path}/checkpoints'

  # model setting
  model_name = 'transformer-translation'
  vocab_num = 22000
  max_length = 512
  d_model = 512
  head_num = 8
  dropout = 0.1
  N = 3
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

  # hyper parameter
  epochs = 5
  batch_size = 16
  padding_idx = tokenizer.pad_token_id
  learning_rate = 0.5

  dataset = TranslationDataset(tokenizer=tokenizer, file_path=data_path, max_length=max_length)
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N)

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

  trainer = TranslationTrainer(dataset, tokenizer, model, max_length, device, model_name, checkpoint_path, batch_size)
  train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

  trainer.train(epochs, train_dataloader, eval_dataloader, optimizer, scheduler)

