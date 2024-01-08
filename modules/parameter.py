import os
import sys
import shutil

sys.path.append("../")
from utils.reader import read_yaml


class Parameter:
    """ Parameter """
    def __init__(self, config):
        """ __init__ """
        self.config_file = config
        self.config = read_yaml(config)
        self.format()
    
    def format(self):
        """ format """
        self.config["trainer"]["experiments"]["exp_dir"] = "/".join([self.config["trainer"]["experiments"]["exp_dir"], 
                                                                     self.config["trainer"]["experiments"]["name"]])
        self.config["trainer"]["experiments"]["weights"] = "/".join([self.config["trainer"]["experiments"]["exp_dir"], 
                                                                     self.config["trainer"]["experiments"]["weights"]])
        self.config["trainer"]["experiments"]["conf"] = "/".join([self.config["trainer"]["experiments"]["exp_dir"], 
                                                                  self.config["trainer"]["experiments"]["conf"]])
        # mkdir experiments
        if not os.path.exists(self.config["trainer"]["experiments"]["exp_dir"]):
            os.mkdir(self.config["trainer"]["experiments"]["exp_dir"])
        if not os.path.exists(self.config["trainer"]["experiments"]["weights"]):
            os.mkdir(self.config["trainer"]["experiments"]["weights"])
        if not os.path.exists(self.config["trainer"]["experiments"]["conf"]):
            os.mkdir(self.config["trainer"]["experiments"]["conf"])
        # copy conf
        shutil.copy(self.config_file, self.config["trainer"]["experiments"]["conf"])
        
    def get_config(self):
        """ get_config """
        return self.config