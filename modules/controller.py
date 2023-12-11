import sys
from accelerate import Accelerator

sys.path.append("../")
from modules.parameter import Parameter
from modules.informer import Informer
from modules.modeller import Modeller
from modules.trainer import Trainer
from modules.analyser import Analyser


class Controller:
    """ The Module to Dispath the Work-Flow """
    def __init__(self, config):
        """ __init__ """
        self.config = Parameter(config).get_config()
        self.accelerator = Accelerator(gradient_accumulation_steps=self.config["accelerator"]["gradient_accumulation_steps"])
        self.informer = Informer(self.config["informer"], self.accelerator)
        self.modeller = Modeller(self.config["modeller"])
        self.analyser = Analyser(self.config["analyser"])
        self.trainer = Trainer(self.config["trainer"], 
                               self.informer, 
                               self.modeller, 
                               self.analyser, 
                               self.accelerator)

    def register(self):
        """ register """
        self.informer.setup()
        self.modeller.setup()
        self.analyser.setup()
        self.trainer.setup()
    
    def dispatch(self):
        """ dispath """
        eval("self.trainer." + self.config["trainer"]["mode"] + "()")

        