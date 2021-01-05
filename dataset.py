from torch.utils.data import Dataset, DataLoader, random_split
from util import load_csv

class TranslationDataset(Dataset):
  def __init__(self, file_path):

    csv_datas = load_csv(file_path)
    self.docs = []
    doc = ""
    while True:
      line = data_file.readline()
      if not line: break

      line = line[:-1]
      if len(tokenizer.encode(doc)) < max_len and len(tokenizer.encode(doc + line)) < max_len:
        doc += line
      elif len(tokenizer.encode(doc + line)) >= max_len and len(tokenizer.encode(doc)) < max_len:
        self.docs.append(doc)
        # print(f"max_len-{max_len} real_len-{len(tokenizer.encode(doc))} doc-{doc}\n\n")
        doc = line
    print('namu wiki data load complete')

  def __len__(self):
    return len(self.documents)

  def __getitem__(self, idx):
    return self.documents[idx]
