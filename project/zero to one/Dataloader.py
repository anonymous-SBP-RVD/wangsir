# Listing 2.5 A dataset for batched inputs and targets
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)                                #A

        for i in range(0, len(token_ids) - max_length, stride):          #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):                                                     #C
        return len(self.input_ids)

    def __getitem__(self, idx):                                            #D
        return self.input_ids[idx], self.target_ids[idx]


#A 将整个文本进行分词
#B 使用滑动窗口将书籍分块为最大长度的重叠序列。
#C 返回数据集的总行数
#D 从数据集中返回指定行
def create_dataloader_v1(txt, batch_size=4, max_length=256,stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                       #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)      #B
    dataloader = DataLoader(
      	dataset,
      	batch_size=batch_size,
      	shuffle=shuffle,
      	drop_last=drop_last,                                        #C
      	num_workers=0                                               #D
    )

    return dataloader

if __name__ == "__main__":
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
