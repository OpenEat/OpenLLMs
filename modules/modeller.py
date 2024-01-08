from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM


class Modeller:
    """ Modeller """
    def __init__(self, config):
        """ __init__ """
        self.config = config
    
    def get_model(self):
        """ get_model """
        return self.model
    
    def get_parameters(self):
        """ get_parameters """
        return self.parameters

    def setup(self):
        """ setup """
        self.model, self.parameters = self.set_model_parameters()
    
    def set_model_parameters(self):
        """ set_model_parameters """
        model, parameters = eval("self." + self.config["mode"] + "()")
        if self.config["gradient_checkpointing_enable"]:
            model.gradient_checkpointing_enable()
        return model, parameters

    def fromscratch(self):
        """ fromscratch """
        model_config = AutoConfig.from_pretrained(self.config["args"]["conf"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(model_config)
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        parameters = [{"params": [p for n, p in model.named_parameters() \
                                  if not any(nd in n for nd in no_decay)],
                      "weight_decay": self.config["weight_decay"]},
                      {"params": [p for n, p in model.named_parameters() \
                                  if any(nd in n for nd in no_decay)], 
                      "weight_decay": 0.0}]
        return model, parameters

    def fparameter(self):
        """ full parameter """
        model_config = AutoConfig.from_pretrained(self.config["args"]["conf"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.config["args"]["ckpt"], 
                                                     config=model_config, 
                                                     trust_remote_code=True)
        model.config.use_cache = False
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        parameters = [{"params": [p for n, p in model.named_parameters() \
                                  if not any(nd in n for nd in no_decay)],
                      "weight_decay": self.config["weight_decay"]},
                      {"params": [p for n, p in model.named_parameters() \
                                  if any(nd in n for nd in no_decay)], 
                      "weight_decay": 0.0}]
        return model, parameters

    def loraeparameter(self):
        """ efficient parameter """
        model = self.fparameter()
        # gradient ckpt bug, https://github.com/huggingface/transformers/issues/23170
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 target_modules=self.config["args"]["target_modules"],
                                 inference_mode=self.config["args"]["inference_mode"],
                                 r=self.config["args"]["rank"],
                                 lora_alpha=self.config["args"]["alpha"],
                                 lora_dropout=self.config["args"]["dropout"])
        model = get_peft_model(model, peft_config)
        parameters = model.parameters()
        return model, parameters
        
        
    