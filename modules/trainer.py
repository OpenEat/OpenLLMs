import os
import sys
import time
import torch
import random
import logging
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import get_cosine_schedule_with_warmup
from transformers.deepspeed import is_deepspeed_zero3_enabled

class Trainer:
    """ Trainer """
    def __init__(self, config, informer, modeller, accelerator):
        """ __init__ """
        self.config = config
        self.informer = informer
        self.modeller = modeller
        self.accelerator = accelerator
        self.log_steps = self.config["log_steps"]
        self.save_steps = self.config["save_steps"]
    
    def setup(self):
        """ setup """
        self.model = self.modeller.get_model()
        self.parameters = self.modeller.get_parameters()
        self.dataloader = self.informer.get_dataloader()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_lr_scheduler()
        self.prepare()

    def set_optimizer(self):
        """ set_optimizer """
        optimizer = FusedAdam(self.parameters,
                              lr=self.config["lr"],
                              betas=(0.9, 0.95))
        return optimizer

    def set_lr_scheduler(self):
        """ set_lr_scheduler """
        training_steps = self.config["epoch"] * self.informer.num_rows // \
                        (self.informer.config["batch_size"] * self.accelerator.num_processes)
        scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.config["warmup_steps"],
                                                    num_training_steps=training_steps // \
                                                        self.accelerator.gradient_accumulation_steps)
        return scheduler

    def prepare(self):
        """ prepare """
        _, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.dataloader, 
                                                                                 self.model, 
                                                                                 self.optimizer, 
                                                                                 self.scheduler)
        self.optimizer.zero_grad()
        self.global_step = 0
        # load pre-checkpoint
        if self.config["experiments"]["load_model"]:
            self.accelerator.load_state(self.config["experiments"]["weights"])
            self.global_step = self.scheduler.scheduler._step_count - 1
            self.global_step = self.global_step // self.accelerator.num_processes
        # skip behind pre-checkpoint's step
        if self.global_step > 0:
            skip_steps = self.global_step * self.accelerator.gradient_accumulation_steps
            logging.warning("Skiped {} steps.".format(skip_steps))
            self.dataloader_skiped = self.accelerator.skip_first_batches(self.dataloader, num_batches=skip_steps)
        else:
            self.dataloader_skiped = self.dataloader
        self.accelerator.wait_for_everyone()
    
    def batch2tensor(self, batch):
        """ batch2tensor """
        min_pad_len = min(batch["pad_len"])
        batch_tensor = {}
        for k, v in batch.items():
            if k not in ["input_ids", "labels"]: continue
            v = v[:, min_pad_len:]
            batch_tensor[k] = v.to(self.accelerator.device, non_blocking=True)
        return batch_tensor
    
    def train_one_step(self, batch):
        """ train_one_step """
        out = self.model(**batch)
        loss = out.loss
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss
    
    def pretrain(self):
        """ train """
        self.model.train()
        self.epoch = 1
        self.step = 1
        self.data_num = 0
        self.start_time = time.time()
        eval_loss = 0.0
        data_nums = []
        while self.epoch <= self.config["epoch"]:
            dataloader = self.dataloader_skiped if self.epoch == 0 else self.dataloader
            for batch in dataloader:
                batch = self.batch2tensor(batch)
                data_nums.append(batch["input_ids"].shape[0])
                # train step
                with self.accelerator.accumulate(self.model):
                    loss = self.train_one_step(batch)
                    eval_loss += loss
                    if self.accelerator.sync_gradients: self.global_step += 1
                # log step
                if self.step > 0 and self.step % self.log_steps == 0:
                    data_nums = torch.tensor(data_nums).to(self.accelerator.device)
                    self.data_num += sum(self.accelerator.gather(data_nums)).item()
                    if self.accelerator.is_main_process:
                        eval_loss /= (self.log_steps)
                        self.log(eval_loss)
                    eval_loss = 0
                    data_nums = []
                # save step
                if self.step > 0 and self.step % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    save_path = self.config["experiments"]["weights"] + "/{}".format(self.step)
                    save_weights(unwrapped_model, save_path)
                self.step += 1
            self.epoch += 1
    
    def sft(self):
        """ sft """
        self.model.train()
        self.epoch = 1
        self.step = 1
        self.data_num = 0
        self.start_time = time.time()
        eval_loss = 0.0
        data_nums = []
        while self.epoch <= self.config["epoch"]:
            dataloader = self.dataloader_skiped if self.epoch == 0 else self.dataloader
            for batch in dataloader:
                batch = self.batch2tensor(batch)
                data_nums.append(batch["input_ids"].shape[0])
                # train step
                with self.accelerator.accumulate(self.model):
                    loss = self.train_one_step(batch)
                    eval_loss += loss
                    if self.accelerator.sync_gradients: self.global_step += 1
                # log step
                if self.step > 0 and self.step % self.log_steps == 0:
                    data_nums = torch.tensor(data_nums).to(self.accelerator.device)
                    self.data_num += sum(self.accelerator.gather(data_nums)).item()
                    if self.accelerator.is_main_process:
                        eval_loss /= (self.log_steps)
                        self.log(eval_loss)
                    eval_loss = 0
                    data_nums = []
                # save step
                if self.step > 0 and self.step % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    save_path = self.config["experiments"]["weights"] + "/{}".format(self.step)
                    save_weights(unwrapped_model, save_path)
                self.step += 1
            self.epoch += 1

    def log(self, losses):
        """ log """
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        tokens = self.informer.config["batch_size"] \
                 * self.log_steps \
                 * self.informer.config["max_seq_length"]
        current_lr = self.optimizer.param_groups[0]["lr"]
        ratio = '%.2f%%' % (100 * self.data_num / self.informer.num_rows)
        self.accelerator.print(
            "Epoch: {}, Global Step: {}, Data Step: {}, Data Process: {} Ratio: {}, Loss: {}, LR: {}, Token per second per gpu: {}".format(
                self.epoch,
                self.global_step,
                self.step,
                str(self.data_num) + "/" + str(self.informer.num_rows),
                ratio,
                losses,
                current_lr,
                tokens / cost_time))            

def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]

def save_weights(model, save_path, zero_stage_3=True):
    """ save_weights """
    os.makedirs(save_path, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_path, WEIGHTS_NAME)
    model_to_save = model.module if hasattr(model, 'module') else model
    if not is_deepspeed_zero3_enabled():
        torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            output_state_dict[k] = v_p
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


