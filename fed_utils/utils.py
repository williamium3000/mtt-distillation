import logging
import torch
import os.path as osp
import torch
import mmcv
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.getLogger().setLevel(logging.INFO)


class Logger(object):
    """record cmd info to file and print it to cmd at the same time
    
    Args:
        log_name (str): log name for output.
        log_file (str): a file path of log file.
    """
    def __init__(self, log_name=None, log_file=None):
        if log_name is not None:
            self.logger = logging.getLogger(log_name)
            self.name = log_name
        else:
            logging.getLogger().setLevel(logging.INFO)
            self.logger = logging
            self.name = "root"

        if log_file is not None:
            handler = logging.FileHandler(log_file, mode='w')
            handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, log_str):
        """Print information to logger"""
        self.logger.info(log_str)

    def warning(self, warning_str):
        """Print warning to logger"""
        self.logger.warning(warning_str)


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k  / batch_size)
        return res


class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, save_path):
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    model = model.module if hasattr(model, 'module') else model
    torch.save(model.state_dict(), save_path)

def load_model(path, model):
    to_load = torch.load(path, map_location="cpu")
    model.load_state_dict(to_load)
    return model

