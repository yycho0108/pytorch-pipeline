#!/usr/bin/env python3

from dataclasses import dataclass
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from pathlib import Path
from simple_parsing import Serializable
import torch as th
import logging

from pytorch_pipeline.train.saver import Saver


class Callback(object):
    """
    Base class for periodic callbacks.
    TODO(ycho): Consider migration to e.g. fast.ai API
    when our custom solution cannot handle some sophisticated function.
    """

    def __init__(self, period: int, callback: Callable[[int], None]):
        self.period = period
        self.callback = callback
        self.last_call = 0

    def on_step(self, step: int):
        # NOTE(ycho): Don't use (step % self.period) which is common,
        # since if `num_env` != 0, lcm(period, num_env) may not be period.
        if step < self.last_call + self.period:
            return
        self.callback(step)
        self.last_call = step


class SaveModelCallback(Callback):
    """ Callback for saving models. """
    @dataclass
    class Settings(Serializable):
        pattern: str = 'save-{:06d}.zip'
        period: int = int(1e3)  # NOTE(ycho): by default, every 1000 steps

    def __init__(self, opts: Settings, path: str, saver: Saver):
        self.opts = opts
        self.path = Path(path)
        self.saver = saver
        super().__init__(self.opts.period, self._save)

    def _save(self, step: int):
        filename = self.path / self.opts.pattern.format(step)
        self.saver.save(filename)


class EvalCallback(Callback):
    """
    Callback for periodic model evaluation.
    Only sets up the boilerplate skeleton code, so `eval_fn` should do the heavy lifting.
    The expected `eval` operations might include:

    * computing relevant metrics (accuracy, recall, ...)
    * logging the relevant metrics (to e.g. TensorBoard, Terminal, ...)
    """

    @dataclass
    class Settings(Serializable):
        period: int = int(1e3)
        num_samples: int = 1

    def __init__(self, opts: Settings, model: th.nn.Module,
                 loader: th.utils.data.DataLoader, eval_fn=None):
        self.opts = opts
        self.model = model
        self.loader = loader
        self.eval_fn = eval_fn
        super().__init__(self.opts.period, self._eval)

    def _eval(self, step: int):
        # NOTE(ycho): Set `model` to `eval` mode, but
        # cache the previous model cfg for restoration.
        print('_eval')
        prev_mode = self.model.training
        self.model.eval()

        # Run evaluation loop ...
        print('<evaluate>')
        count = 0
        with th.no_grad():
            for data in self.loader:
                # Eval step...
                self.eval_fn(step, self.model, data)

                # Increment eval counts
                count += 1
                if count >= self.opts.num_samples:
                    break
        print('</evaluate>')

        # NOTE(ycho): Restore previous training mode.
        self.model.train(prev_mode)


class Callbacks(Callback):
    """ Class to deal with a bunch of callbacks """

    def __init__(self, callbacks: Optional[List[Callback]] = []):
        self.callbacks = callbacks

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def on_step(self, step: int):
        for cb in self.callbacks:
            cb.on_step(step)
