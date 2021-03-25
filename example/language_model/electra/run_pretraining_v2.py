import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../') # for local
# sys.path.append('/content/drive/My Drive/Colab Notebooks/transformer-electra') # for colab

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from transformers import BertTokenizer
from transformers.optimization import AdamW
import os
import json
import logging
from datetime import datetime
from model.electra_v2 import Electra
from example.language_model.common.arg import ElectraConfig
from example.language_model.common.dataset import DatasetForMLM


class ElectraTrainer(object):
    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 max_len,
                 model_name,
                 checkpoint_path,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 log_dir='../logs'):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.log_dir = log_dir

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=eval_shuffle)
        logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                         eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')

        return train_loader, eval_loader

    def train(self,
              epochs,
              train_dataloader,
              eval_dataloader,
              optimizer,
              scheduler,
              log_steps,
              ckpt_steps,
              gradient_accumulation_steps=1):

        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0
        start_epoch = 0
        start_step = 0

        # Load Checkpoint
        if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
            checkpoint = torch.load(f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            global_steps = checkpoint['train_step']
            start_step = global_steps if start_epoch == 0 else global_steps % len(train_dataloader)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.model.train()
        self.model.to(self.device)

        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
        logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        self.model.zero_grad()  # Reset gradients tensors
        for epoch in range(start_epoch, epochs): #tqdm(range(epochs), desc='Epochs', position=0):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            pb = tqdm(enumerate(train_dataloader),
                      desc=f'Epoch-{epoch} Iterator',
                      total=len(train_dataloader),
                      bar_format='{l_bar}{bar:10}{r_bar}'
                      )
            for step, batch in pb:
                if step < start_step:
                    continue
                inputs, input_mask, labels = batch  # _ is input_mask
                inputs, input_mask, labels = inputs.to(self.device), input_mask.to(self.device), labels.to(self.device)
                output = self.model(input=inputs, input_mask=input_mask, mlm_label = labels)

                loss = output.loss
                origin_loss = loss.item()

                loss = loss / gradient_accumulation_steps  # divide loss into gradient accumulation step
                loss.backward()

                step_loss += origin_loss
                losses[global_steps] = origin_loss

                local_steps += 1
                global_steps += 1

                if global_steps % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scheduler.step()
                    optimizer.step()
                    self.model.zero_grad()

                if global_steps % log_steps == 0:
                    pb.set_postfix_str(f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps} | disc_acc: {output.disc_acc} | gen_acc: {output.gen_acc}''')
                    step_loss = 0.0
                    local_steps = 0

                if global_steps % ckpt_steps == 0:
                    self.save(epoch, self.model, optimizer, scheduler, losses, global_steps)
                    logging.info(f'{datetime.now()} | Saved checkpoint to: {self.checkpoint_path}')
                    with open(f'{self.log_dir}/{self.model_name}_train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()

            # Evaluate every epoch
            self.evaluate(eval_dataloader)
            self.model.train()
            start_step = 0

        self.save(epoch, self.model, optimizer, scheduler, losses, global_steps)

        return self.model

    def evaluate(self, dataloader):
        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        eval_loss = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader),
                                desc='Evaluating',
                                leave=True,
                                total=len(dataloader),
                                bar_format='{l_bar}{bar:10}{r_bar}'):
            inputs, input_mask = batch
            inputs, input_mask= inputs.to(self.device), input_mask.to(self.device)

            with torch.no_grad():
                output = self.model(inputs, input_mask)

            tmp_eval_loss = output.loss

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            eval_steps += 1

            total_eval_loss = eval_loss/eval_steps

            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss}')
            with open(f'{self.log_dir}/{self.model_name}_eval_results.txt', 'a+') as results_file:
                results_file.write(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss}\n')
                results_file.close()

    def save(self, epoch, model, optimizer, scheduler, losses, train_step):
        torch.save({
            'epoch': epoch,                                 # 현재 학습 epoch
            'model_state_dict': model.state_dict(),         # 모델 저장
            'optimizer_state_dict': optimizer.state_dict(), # 옵티마이저 저장
            'scheduler_state_dict': scheduler.state_dict(), # 스케줄러
            'losses': losses,                               # Loss 저장
            'train_step': train_step,                       # 현재 진행한 학습
        }, f'{self.checkpoint_path}/{self.model_name}.pth')


def main():
    torch.manual_seed(9)
    # base_path = '/content/drive/My Drive/Colab Notebooks/transformer'
    base_path = '/Users/a60058238/Desktop/dev/workspace/transformers'
    log_dir = f'{base_path}/logs'
    config_path = f'{base_path}/config/electra/electra-train.json'

    # 1. Config
    train_config, gen_config, disc_config = ElectraConfig(config_path=config_path).get_config()

    # 2. Tokenizer
    tokenizer = BertTokenizer(vocab_file=train_config.vocab_path, do_lower_case=False)

    # 3. Dataset
    dataset = DatasetForMLM(tokenizer, train_config.max_seq_len, path=train_config.data_path)

    # 4. Electra Model
    model = Electra(
        config=train_config,
        gen_config=gen_config,
        disc_config=disc_config,
        num_tokens=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,          # the token id reserved for masking
        pad_token_id=tokenizer.pad_token_id,            # the token id for padding
        mask_prob=0.15,                                 # masking probability for masked language modeling
        mask_ignore_token_ids=tokenizer.all_special_ids # ids of tokens to ignore for mask modeling ex. (cls, sep)
    )

    # weight tie any other embeddings if available, token type embeddings, etc.
    model.tie_embedding_weight()
    # model.cuda()

    # 5. Pretraining Traniner
    trainer = ElectraTrainer(dataset, model, tokenizer,
                             model_name=train_config.model_name,
                             max_len=train_config.max_seq_len,
                             checkpoint_path=train_config.checkpoint_path,
                             train_batch_size=train_config.batch_size,
                             eval_batch_size=train_config.batch_size,
                             log_dir=log_dir)

    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    learning_rate = 5e-4
    adam_epsilon = 1e-6

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,                       # Optimzer
                                                step_size=len(train_dataloader), # Gamma 비율로 줄일 스텝사이즈
                                                gamma=0.9)                       # lr줄이는 비율
    """
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=0.1, 
                                                steps_per_epoch=len(train_dataloader)//train_config.gradient_accumulation_steps, 
                                                epochs=train_config.epochs,
                                                pct_start = 0.1
                                                anneal_strategy='linear')
    """

    trainer.train(epochs=train_config.epochs,
                  train_dataloader=train_dataloader,
                  eval_dataloader=eval_dataloader,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  log_steps=train_config.log_steps,
                  ckpt_steps=train_config.ckpt_steps,
                  gradient_accumulation_steps=train_config.gradient_accumulation_steps)

if __name__ == '__main__':
    main()
