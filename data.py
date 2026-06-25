import os
import torch
from utils import *
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random


def load_split_data(config):
    def transform_token2id_seq(token_sequences, item_token_to_id):
        id_sequences = []
        for interaction_record in token_sequences:
            item_token_sequence = interaction_record["inter_history"]
            item_id_sequence = [item_token_to_id[token] for token in item_token_sequence]
            target_id = item_token_to_id[interaction_record["target_id"]]
            id_sequences.append(item_id_sequence + [target_id])

        return id_sequences
            
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, f"{dataset}/{dataset}")
    map_path = dataset_path + config["map_path"]
    
    train_inter = load_jsonl(dataset_path + ".train.jsonl")
    valid_inter = load_jsonl(dataset_path + ".valid.jsonl")
    test_inter = load_jsonl(dataset_path + ".test.jsonl")

    item_token_to_id = load_json(map_path) # id start from 1, 2, ...
    
    train_sequences = transform_token2id_seq(train_inter, item_token_to_id)
    valid_sequences = transform_token2id_seq(valid_inter, item_token_to_id)
    test_sequences = transform_token2id_seq(test_inter, item_token_to_id)

    
    n_items = len(item_token_to_id)

    return item_token_to_id, n_items, train_sequences, valid_sequences, test_sequences
    
    
class SequentialSplitDataset(Dataset):
    def __init__(self, config, n_items, inter_seq, data_ratio=1):
        self.n_items = n_items
        self.config = config

        if data_ratio < 1:
            # random sampling
            sample_count = int(len(inter_seq)*data_ratio)
            inter_seq = random.sample(inter_seq, sample_count)
            
        self.data = self.__map_inter__(inter_seq)

    def __map_inter__(self, inter_seq):
        mapped_samples = []

        for sequence in inter_seq:
            target = sequence[-1]
            sample = {"id_seq": sequence[:-1], "target": [target]}
            mapped_samples.append(sample)

        return mapped_samples
            
    def __getitem__(self, idx):
        sample = self.data[idx]
        id_seq = sample['id_seq']
        target = sample['target']
        
        return id_seq, target

    def __len__(self):
        return len(self.data)
    
    
class Collator(object):
    def __init__(self, eos_token_id, pad_token_id, max_length):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __pad_seq__(self, seq):
        if len(seq) > self.max_length:
            return seq[-self.max_length+1:]
        return seq
    
    def __call__(self, batch):
        id_sequences, targets = zip(*batch)
        
        input_ids = [torch.tensor(self.__pad_seq__(id_sequence)) for id_sequence in id_sequences]
        input_ids = pad_sequence(input_ids).transpose(0, 1)
        input_ids = input_ids.to(torch.long)

        attention_mask = (input_ids != self.pad_token_id).bool()
        
                              
        targets = torch.tensor(targets)

        targets = targets.to(torch.long).contiguous()
        
        return dict(input_ids=input_ids,
                    attention_mask=attention_mask,
                    targets=targets)


