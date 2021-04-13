import torch
from torch.utils.data import Dataset, DataLoader, random_split
from util import load_csv
from model.util import subsequent_mask
from transformers import BertTokenizer
from torch.autograd import Variable
from tqdm import tqdm

class TranslationDataset(Dataset):
  def __init__(self, tokenizer:BertTokenizer, file_path:str, max_length:int):
    pad_token_idx = tokenizer.pad_token_id
    csv_datas = load_csv(file_path)
    self.docs = []
    # for line in csv_datas: # line[0] 한글, line[1] 영어
    for line in tqdm(csv_datas):
      input = tokenizer.encode(line[0],max_length=max_length,truncation=True)
      rest = max_length - len(input)
      input = torch.tensor(input + [pad_token_idx]*rest)

      target = tokenizer.encode(line[1], max_length=max_length, truncation=True)
      rest = max_length - len(target)
      target = torch.tensor(target+ [pad_token_idx] * rest)

      doc={
        'input_str': tokenizer.convert_ids_to_tokens(input),
        'input':input,                                        # input
        'input_mask': (input != pad_token_idx).unsqueeze(-2),       # input_mask
        'target_str': tokenizer.convert_ids_to_tokens(target),
        'target': target,                                       # target,
        'target_mask': self.make_std_mask(target, pad_token_idx),    # target_mask
        'token_num': (target[...,1:] != pad_token_idx).data.sum()  # token_num
      }
      self.docs.append(doc)
  @staticmethod
  def make_std_mask(tgt, pad_token_idx):
    'Create a mask to hide padding and future words.'
    target_mask = (tgt != pad_token_idx).unsqueeze(-2)
    target_mask = target_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(target_mask.data))
    return target_mask.squeeze()

  def __len__(self):
    return len(self.docs)
  def __getitem__(self, idx):
    item = self.docs[idx]
    return item