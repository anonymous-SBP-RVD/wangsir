# Listing 7.4 Implementing an instruction dataset class
import torch
from torch.utils.data import Dataset
import tiktoken
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:                                           #A
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)          #A
    inputs_lst = []

    for item in batch:                                             #B
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])                         #C
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)             #D
    return inputs_tensor


def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])             #A
        targets = torch.tensor(padded[1:])             #B
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

#A 找出批量中的最长序列
#B 对输入进行填充并准备好输入数据
#C 删除之前添加的多余填充 token
#D 将输入列表转换为张量，并转移到目标设备

# Listing 7.5 Implementing a custom batch collate function
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1]) # Truncate the last token for inputs
        targets = torch.tensor(padded[1:]) # Shift +1 to the right for targets

        mask = targets == pad_token_id                  #A
        indices = torch.nonzero(mask).squeeze()         #A
        if indices.numel() > 1:                         #A
            targets[indices[1:]] = ignore_index         #A

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]        #B
            targets = targets[:allowed_max_length]      #B

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst)
    return inputs_tensor, targets_tensor

#A 在 targets 中，将除第一个以外的所有填充标记替换为 ignore_index
# cross_entropy 函数的默认设置是 cross_entropy(..., ignore_index=-100)，这意味着它会忽略标签为 -100 的目标。
#B 可选择性地将序列截断到最大长度


