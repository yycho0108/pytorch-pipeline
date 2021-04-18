#!/usr/bin/env python3

__all__ = ["Trainer"]

from dataclasses import dataclass
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from pathlib import Path
from simple_parsing import Serializable
import torch as th
import logging

from pytorch_pipeline.train.callback import Callbacks
from pytorch_pipeline.train.saver import Saver


class Trainer(object):
    """
    Generic trainer for a pytorch nn.Module.
    Intended to be flexible, modify as needed.
    """

    @dataclass
    class Settings(Serializable):
        train_steps: int = int(1e4)
        eval_period: int = int(1e3)
        num_epochs: int = int(1)

    def __init__(self,
                 opts: Settings,
                 model: th.nn.Module,
                 optimizer: th.optim.Optimizer,
                 loss_fn: Callable[[th.nn.Module, Any], th.Tensor],
                 callbacks: Callbacks,
                 loader: th.utils.data.DataLoader
                 ):
        self.opts = opts
        self.model = model
        self.optim = optimizer

        # NOTE(ycho): loss_fn != th.nn.*Loss
        # since it directly takes `model` as argument.
        # This is (for now) intended to support maximum flexibility.
        # The simplest such version might be something like:
        # def loss_fn(model, data):
        #     x, y = data
        #     y_ = model(x)
        #     return th.square(y_-y).mean()

        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.loader = loader

    def train(self):
        self.model.train()

        step = 0
        # TODO(ycho): Consider registering more hooks.
        #self.callbacks.on_train(train)
        try:
            for epoch in range(self.opts.num_epochs):
                # self.callbacks.on_epoch(epoch)
                for i, data in enumerate(self.loader):
                    # Compute loss ...
                    # NOTE(ycho): if one of the callbacks require training loss,
                    # e.g. for logging, simply register a hook to the loss module
                    # rather than trying to extract them here.
                    loss = self.loss_fn(self.model, data)

                    # Backprop + Optimize ...
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # Deal with callbacks ...
                    # == logging, saving, evaluation
                    self.callbacks.on_step(step)
                    step += 1

        except KeyboardInterrupt:
            logging.info('Terminating training due to SIGINT')
        finally:
            # TODO(ycho): When an interrupt occurs, the current state
            # will ALWAYS be saved to a hardcoded file in a temporary directory.
            # Maybe this is a bad idea.
            Saver(self.model, self.optim).save('/tmp/model-backup.zip')
