import os
import sys
from glob import glob
from datasets import load_dataset

sys.path.append("../")
from utils.reader import read_text


class Analyser:
    """ Analyser """
    def __init__(self, config):
        """ __init__ """
        self.config = config
    
    def setup(self):
        """ setup """
        self.set_tokenzier()
        self.set_task()
    
    def set_tokenzier(self):
        """ """
        tokenizer = AutoTokenizer.from_pretrained(self.config["tokenzier"]["path"],
                                                  pad_token=self.config["tokenzier"]["special_tokens"]["pad_token"],
                                                  eos_token=self.config["tokenzier"]["special_tokens"]["eos_token"],
                                                  trust_remote_code=True)
        return tokenizer

    def set_task(self):
        """ set_task """
        task_infos = {}
        for task in self.config["task"]:
            task_infos[task["function"]] = self.format(**task)
        return task_infos

    def format(self, function, data):
        """ format """
        data_infos = {}
        for name, pattern in data.items():
            data_files = glob(pattern)
            dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)
            dataset.map(lambda x: self.tokenizer(x[""]))
            data_infos[name] = dataset
    
    

