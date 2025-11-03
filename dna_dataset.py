import numpy as np
import torch
from torch.utils.data import Dataset
import os


class DNADatasetOH(Dataset):
    def __init__(self, data_path, seq_length=510, vocab_size=9, split="train", temperature=1.0):
        """
        :param data_path: path, The one-hot encoded data of shape (L, V)
        :param seq_length: The length of the segment to extract
        :param vocab_size: The size of the vocabulary (V, )
        """
        if split == "test":
            data_shape = (55982971, vocab_size)
            ratio = 1.0

        elif split == "train":
            data_shape = (2349430289, vocab_size)  # 2872048919, multi: 2895685324
            ratio = 1.2
        else:
            raise NotImplementedError(f"No '{split}' dataset, please check again.")

        self.data = np.memmap(os.path.join(data_path, f"{split}_data.npy"), dtype=np.float32,
                              mode='r', shape=data_shape)

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.total = int(data_shape[0] / self.seq_length * ratio)  
        self.max_start_idx = data_shape[0] - self.seq_length - 1

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        length = self.seq_length + 2
        input_ids = torch.zeros((length, self.vocab_size), dtype=torch.float32)
        input_ids[0, 1] = 1  # [CLS]: 1
        input_ids[-1, 2] = 1  # [SEP]: 2

        for _ in range(10):
            start_idx = np.random.randint(0, self.max_start_idx)
            mmap_squeue = self.data[start_idx: start_idx + self.seq_length]
            
            if np.sum(mmap_squeue[:, 2]) == 0.:
                # 2: <sep> token
                break

        sample = torch.from_numpy(mmap_squeue.astype(np.float32))
        
        if self.temperature != 1.0:
            # temp scaling
            mask = sample != 0
            masked_sample = torch.where(mask, sample, torch.tensor(float('-inf')))
            sample = torch.nn.functional.softmax(masked_sample / self.temperature, dim=-1)

        input_ids[1:-1] = sample  # tokenized, smooth input_ids

        attention_mask = torch.ones(length, dtype=torch.int64)
        tokenized_sample = {"input_ids": input_ids,
                            "attention_mask": attention_mask
                            }
        return tokenized_sample