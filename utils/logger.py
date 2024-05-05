import os
import sys
import torch
import logging
import functools
import pandas as pd
from numpy import nan
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'cur_total', 'cur_counts'])
        self.keys = keys
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def reset_batch(self):
        self._data.cur_total.values[:] = 0
        self._data.cur_counts.values[:] = 0

    def update(self, key, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.cur_total[key] += value * n
        self._data.cur_counts[key] += n

    def avg(self, key):
        if self._data.counts[key] == 0:
            return nan
        return self._data.total[key] / self._data.counts[key]

    def avg_batch(self, key):
        if self._data.cur_counts[key] == 0:
            return nan
        return self._data.cur_total[key] / self._data.cur_counts[key]

    def result(self):
        return {k: self.avg(k) for k in self._data.index}

    def log(self, logger, epoch, train=True):
        message = "Train " if train else "Test "
        if epoch is not None:
            message += 'epoch: %3d ' % (epoch,)
        for k, v in self.result().items():
            message += ' %s: %.4f ' % (k, v)
        logger.info(message)

    def __getitem__(self, key):
        return self.avg(key)
