import warnings
warnings.filterwarnings("ignore")

import os
import torch
from torch.autograd import Variable
from model.util import subsequent_mask
from model.transformer import Transformer
from transformers import BertTokenizer

if __name__=="__main__":
  # project_dir = '/content/drive/My Drive/Colab Notebooks/transformer/'
  project_dir = '../..'
  vocab_path = f'{project_dir}/data/wiki-vocab.txt'
  data_path = f'{project_dir}/data/ko-en-translation.csv'
  checkpoint_path = f'{project_dir}/checkpoints'

  # model setting
  # model_name = 'transformer-translation-spoken'
  model_name = 'transformer-translation-spoken'
  vocab_num = 22000
  max_length = 64
  d_model = 512
  head_num = 8
  dropout = 0.1
  N = 6
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
  model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N)
  if os.path.isfile(f'{checkpoint_path}/{model_name}.pth'):
    checkpoint = torch.load(f'{checkpoint_path}/{model_name}.pth', map_location=device)
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    global_steps = checkpoint['train_step']

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'{checkpoint_path}/{model_name}-.pth loaded')
    model.eval()
  while True:
    # input_str = 'c는 결혼은 자신의 삶을 경제적으로 안정되게 할 수단이라고 생각해요.'#
    input_str = input('입력: ')
    str = tokenizer.encode(input_str)
    pad_len = (max_length - len(str))
    str_len = len(str)
    encoder_input = torch.tensor(str + [tokenizer.pad_token_id]* pad_len)
    encoder_mask = (encoder_input != tokenizer.pad_token_id).unsqueeze(0)

    target = torch.ones(1, 1).fill_(tokenizer.cls_token_id).type_as(encoder_input)
    encoder_output = model.encode(encoder_input,encoder_mask)
    for i in range(max_length - 1):
      lm_logits = model.decode(encoder_output,encoder_mask,target, Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data)))
      prob = lm_logits[:, -1]
      _, next_word = torch.max(prob, dim=1)
      # print(f'ko: {input_str} en: {tokenizer.decode(target.squeeze().tolist(), skip_special_tokens=True)}')

      if next_word.data[0] == tokenizer.pad_token_id or next_word.data[0] == tokenizer.sep_token_id:
        print(f'ko: {input_str} en: {tokenizer.decode(target.squeeze().tolist(),skip_special_tokens=True)}')
        break
      target = torch.cat((target[0], next_word))
      target = target.unsqueeze(0)



