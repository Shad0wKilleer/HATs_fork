import os
import argparse
import torch
import torch.distributed as dist

from utils_engine.logger import get_logger
from utils_engine.pyt_utils import all_reduce_tensor

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex .")

logger = get_logger()


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info("PyTorch Version {}".format(torch.__version__))
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.args = self.parser.parse_args()

        if "WORLD_SIZE" in os.environ:
            self.distributed = int(os.environ["WORLD_SIZE"]) > 1
            print("WORLD_SIZE is %d" % (int(os.environ["WORLD_SIZE"])))

        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.devices = [i for i in range(self.world_size)]
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices = [i for i in range(len(gpus.split(",")))]

    def data_parallel(self, model):
        if self.distributed:
            model = DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model)
        return model

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return torch.mean(tensor)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process"
            )
            return False
