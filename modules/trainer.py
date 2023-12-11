import os
import sys
import time
import torch
import random
import logging
from deepspeed.ops.adam import FusedAdam
from transformers import get_cosine_schedule_with_warmup

class Trainer:
    """ Trainer """
    def __init__(self, config, informer, modeller, analyser, accelerator):
        """ __init__ """
        self.config = config
        self.informer = informer
        self.modeller = modeller
        self.analyser = analyser
        self.accelerator = accelerator
        self.lr_scheduler_factor = accelerator.num_processes / accelerator.gradient_accumulation_steps
        self.log_steps = self.config["log_steps"] / accelerator.gradient_accumulation_steps
        self.eval_steps = self.config["eval_steps"] / accelerator.gradient_accumulation_steps
        self.save_steps = self.config["save_steps"] / accelerator.gradient_accumulation_steps
    
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
        scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.config["warmup_steps"] * self.lr_scheduler_factor,
                                                    num_training_steps=self.config["warmup_steps"] * self.lr_scheduler_factor)
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
        losses = {"total_loss": loss}
        return losses
    
    def pretrain(self):
        """ train """
        self.epoch = 0
        self.step = 0
        self.model.train()
        self.start_time = time.time()
        while True:
            dataloader = self.dataloader_skiped if self.epoch == 0 else self.dataloader
            for batch in dataloader:
                batch = self.batch2tensor(batch)
                # train step
                with self.accelerator.accumulate(self.model):
                    losses = self.train_one_step(batch)
                    if self.accelerator.sync_gradients: self.global_step += 1
                # log step
                if self.step > 0 and self.step % self.log_steps == 0 \
                   and self.accelerator.is_main_process:
                    self.log(losses)
                # eval step
                    #TODO 
                # save step
                if self.step > 0 and self.step % self.save_step == 0:
                    self.accelerator.save_state(self.work_dir)
                self.step += 1
            self.epoch += 1
    
    def sft(self):
        """ sft """
        # TODO
        pass

    def log(self, losses):
        """ log """
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        tokens = self.informer.config["batch_size"] \
                 * self.log_steps \
                 * self.informer.config["max_seq_length"]
        # wandb.log({"Training/Token per second per gpu": tokens / cost_time})
        current_lr = self.optimizer.param_groups[0]["lr"]
        # wandb.log({"Training/LR": current_lr})
        self.accelerator.print(
            "Epoch: {}, Global Step: {}, Data Step: {}, Loss: {}, LR: {}, Token per second per gpu: {}".format(
                self.epoch,
                self.global_step,
                self.step,
                losses["total_loss"],
                current_lr,
                tokens / cost_time))            
