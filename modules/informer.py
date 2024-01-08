import random
from glob import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node

random.seed(42)


class Informer:
    """ Informer """
    def __init__(self, config, accelerator):
        """ ___init__ """
        self.config = config
        self.accelerator = accelerator
    
    def setup(self):
        """ setup """
        # tokenzier
        self.tokenizer = self.set_tokenzier()
        self.dataset = self.set_dataset(world_size=self.accelerator.num_processes)
        self.dataloader = self.set_dataloader()

    def set_tokenzier(self):
        """ set_tokenzier """
        tokenizer = AutoTokenizer.from_pretrained(self.config["tokenzier"]["path"],
                                                  pad_token=self.config["tokenzier"]["special_tokens"]["pad_token"],
                                                  eos_token=self.config["tokenzier"]["special_tokens"]["eos_token"],
                                                  trust_remote_code=True)
        return tokenizer

    def set_dataset(self):
        """ get_datasets """
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
        # process by different mode
        dataset = self.convert(dataset)
        # padding to max length
        dataset = self.padding(dataset)
        # cal pad length
        dataset = self.padlengthing(dataset)
        # add labels
        dataset = self.labeling(dataset)
        dataset = dataset.shuffle(seed=42)
        # split dataset to different node
        dataset = split_dataset_by_node(dataset, 
                                        rank=self.accelerator.process_index,
                                        world_size=self.accelerator.num_processes)
        return dataset
    
    def convert(self, dataset):
        """ convert """
        dataset = eval("dataset.map(" + self.config["mode"] + ", batched=True, batch_size=1)")
        return dataset
    
    def padding(self, dataset):
        """ padding """
        dataset = dataset.map(lambda x: {"tokenize": self.tokenizer(x["text"],
                                                     return_tensors="pt",
                                                     return_attention_mask=False,
                                                     padding="max_length",
                                                     max_length=self.config["max_seq_length"],
                                                     truncation=True)})
        dataset = dataset.map(lambda x: {"input_ids": x["tokenize"]["input_ids"][0]})
        dataset = dataset.select_columns("input_ids")
        return dataset
    
    def labeling(self, dataset):
        """ labeling """
        dataset = dataset.map(get_labels_gen(self.tokenizer.pad_token_id))
        return dataset
    
    def padlengthing(self, dataset):
        """ padlengthing """
        dataset = dataset.map(get_pad_len(self.tokenizer.pad_token_id))
        return dataset

    def set_dataloader(self):
        """ get_generater """
        dataloader = DataLoader(self.dataset, 
                                batch_size=self.config["batch_size"],
                                num_workers=self.config["num_workers"],
                                prefetch_factor=self.config["prefetch_factor"],
                                pin_memory=True)
        return dataloader
    
    def get_dataloader(self):
        """ get_dataloader """
        return self.dataloader

def get_labels_gen(pad_token_id):
    def get_labels(line):
        input_ids = line["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100
        return {"labels": labels}
    return get_labels

def get_pad_len(pad_token_id):
    """ get_seq_len """
    def pad_length(line):
        input_ids = line["input_ids"]
        padded_len = len(input_ids)
        pad_len = 0
        for id in input_ids:
            if id == pad_token_id:
                pad_len += 1
            else:
                break
        return {"pad_len": pad_len}
    return pad_length

def pretrain(text):
    """ pretrain """
    return text