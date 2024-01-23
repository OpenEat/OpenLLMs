import sys
import random
from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
from transformers.trainer_pt_utils import LabelSmoother

sys.path.append("../")
from utils.data import *

random.seed(42)


class BasicInformer:
    """ Informer """
    def __init__(self, config, accelerator):
        """ __init__ """
        self.config = config
        self.accelerator = accelerator
    
    def setup(self):
        """ setup """
        # tokenzier
        self.tokenizer = self.set_tokenzier()
        self.dataset = self.set_dataset()
        self.dataset = self.process_dataset()
    
    def set_tokenzier(self):
        """ set_tokenzier """
        tokenizer = AutoTokenizer.from_pretrained(self.config["tokenzier"]["path"],
                                                  pad_token=self.config["tokenzier"]["special_tokens"]["pad_token"],
                                                  eos_token=self.config["tokenzier"]["special_tokens"]["eos_token"],
                                                  trust_remote_code=True)
        return tokenizer

    def set_dataset(self):
        """ set_dataset """
        # read data files
        data_files = []
        for name, data_infos in self.config["data"].items():
            pattern = data_infos["pattern"]
            sub_data_files = glob(pattern)
            random.shuffle(sub_data_files)
            use_num = int(len(sub_data_files) * data_infos["ratio"])
            if use_num != 0:
                data_files.extend(sub_data_files[: use_num])
        random.shuffle(data_files)
        dataset = load_dataset("json", 
                               split="train", 
                               data_files=data_files,
                               download_mode="force_redownload")
        self.num_rows = dataset.num_rows
        dataset = dataset.to_iterable_dataset(num_shards=self.config["num_shards"])
        dataset = dataset.shuffle(seed=42)
        return dataset

    def get_dataloader(self):
        """ get_generater """
        dataloader = DataLoader(self.dataset, 
                                batch_size=self.config["batch_size"],
                                num_workers=self.config["num_workers"],
                                prefetch_factor=self.config["prefetch_factor"],
                                pin_memory=True)
        return dataloader

    def process_dataset(self):
        """ process_dataset """
        return self.dataset

            
class PretrainInformer(BasicInformer):
    """ PretrainInformer """
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
    
    def process_dataset(self):
        """ process_dataset """
        # covert sft to pretrain
        self.convert()
        # padding to max length
        self.padding()
        # cal pad length
        self.padlengthing()
        # add labels
        self.labeling()

    def convert(self):
        """ convert """
        self.dataset = self.dataset.map(sft2pretrain)

    def padding(self):
        """ padding """
        self.dataset = self.dataset.map(lambda x: {"tokenize": self.tokenizer(x["text"],
                                                               return_tensors="pt",
                                                               return_attention_mask=False,
                                                               padding="max_length",
                                                               max_length=self.config["max_seq_length"],
                                                               truncation=True)})
        self.dataset = self.dataset.map(lambda x: {"input_ids": x["tokenize"]["input_ids"][0]})
        self.dataset = self.dataset.select_columns("input_ids")
    
    def labeling(self):
        """ labeling """
        self.dataset = self.dataset.map(get_labels_gen(self.tokenizer.pad_token_id))
    
    def padlengthing(self):
        """ padlengthing """
        self.dataset = self.dataset.map(get_pad_len(self.tokenizer.pad_token_id))


class SFTInformer(BasicInformer):
    """ SFTInformer """
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)

    def process_dataset(self):
        """ process_dataset """
        # covert sft to chat pair
        self.covert()
        # padding to max length
        self.padding()
        # cal pad length
        self.padlengthing()
        # add labels
        self.labeling()
    
    def covert(self):
        """ covert """
        self.dataset = self.dataset.map(sft2pair)
    
    def padding(self):
        """ padding """
        def preprocess(data):
            """ preprocess """
            chats = data["text"]
            input_ids = []
            targets_mask = []
            for chat in chats:
                prompt = 'user\n' + chat["user"]
                completion = "assistant\n" + chat["assistant"]
                prompt = self.tokenizer.im_start_id + self.tokenizer(prompt).input_ids + self.tokenizer.im_end_id + self.tokenizer("\n").input_ids
                completion = self.tokenizer.im_start_id + self.tokenizer(completion).input_ids + self.tokenizer.im_end_id + self.tokenizer("\n").input_ids
                input_ids += prompt + completion
                targets_mask += [0] * len(prompt) + [1] * len(completion)
            if len(input_ids) > self.config["max_seq_length"]:
                input_ids = input_ids[:self.config["max_seq_length"]]
                targets_mask = targets_mask[:self.config["max_seq_length"]]
            else:
                input_ids = [self.tokenizer.pad_token_id] * (self.config["max_seq_length"] - len(input_ids)) + input_ids
                targets_mask = [self.tokenizer.pad_token_id] * (self.config["max_seq_length"] - len(targets_mask)) + targets_mask
            data["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
            data["targets_mask"] = torch.tensor(targets_mask, dtype=torch.int64)
            return data
        self.dataset = self.dataset.map(preprocess)
    
    def padlengthing(self):
        """ padlengthing """
        self.dataset = self.dataset.map(get_pad_len(self.tokenizer.pad_token_id))

    def labeling(self):
        """ labeling """
        self.dataset = self.dataset.map(get_labels_gen(self.tokenizer.pad_token_id))

class Informer:
    """ Informer """
    global INFORMERS
    INFORMERS = {"pretrain": PretrainInformer, "sft": SFTInformer}
    def __new__(self, config, accelerator):
        return INFORMERS[config["mode"]](config, accelerator)