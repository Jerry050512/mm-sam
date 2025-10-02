#!/usr/bin/env python
import argparse
import os
import random
import sys
import torch
import numpy as np
from os.path import join

from utilbox.parse_utils import str2list, str2bool, str2dict
from utilbox.train_managers.base import TrainManager
from utilbox.yaml_utils import load_yaml
from utilbox.global_config import PROJECT_ROOT, EXP_ROOT

def set_random_seeds(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_experiment(args: argparse.Namespace, device: torch.device):
    """create a training manager to run the experiment (train or test)"""
    train_manager = TrainManager(device=device, args=args)
    if not args.test_only:
        train_manager.train()
    if not args.train_only:
        train_manager.test()

class Launcher:
    worker_fn: callable = run_experiment

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        group = parser.add_argument_group("Shared Arguments")
        group.add_argument('--config_name', default=None, type=str)
        group.add_argument('--seed', default=3407, type=int)
        group.add_argument('--use_amp', default=False, type=str2bool)
        group.add_argument('--train_only', default=False, type=str2bool)
        group.add_argument('--test_only', default=False, type=str2bool)
        group.add_argument('--train_agent', default=None, type=str)
        group.add_argument('--agent_kwargs', default={}, type=str2dict)

        group = parser.add_argument_group("Training Arguments")
        group.add_argument('--train_bs', default=None, type=int)
        group.add_argument('--train_workers_per_bs', default=0.5, type=float)
        group.add_argument('--train_epoch_num', default=None, type=int)
        group.add_argument('--valid_bs', default=1, type=int)
        group.add_argument('--valid_workers_per_bs', default=0.5, type=float)

        group = parser.add_argument_group("Testing Arguments")
        group.add_argument('--ckpt_path', type=str, default='checkpoints/')
        group.add_argument('--test_bs', default=1, type=int)
        group.add_argument('--test_workers_per_bs', default=0.5, type=float)
        return parser

    @classmethod
    def configure(cls) -> argparse.Namespace:
        args = cls.get_parser().parse_args()
        assert PROJECT_ROOT is not None, "Please register PROJECT_ROOT in utilbox/global_config.py!"
        config_yaml_path = f"{PROJECT_ROOT}/config/{args.config_name}"
        if not config_yaml_path.endswith('.yaml'):
            config_yaml_path += '.yaml'
        config = load_yaml(config_yaml_path)

        known_args_list = [item.dest for item in cls.get_parser()._actions]
        for k, v in config.items():
            if k in known_args_list:
                setattr(args, k, v)
            else:
                raise ValueError(f"Unknown argument '{k}' in your .yaml file!")

        launch_name_list = args.config_name.split('/')
        if launch_name_list[-1].endswith('.yaml'):
            launch_name_list[-1] = launch_name_list[-1][:-len('.yaml')]
        launch_name = '/'.join(launch_name_list)
        args.launch_folder_path = join(EXP_ROOT, launch_name)
        os.makedirs(args.launch_folder_path, exist_ok=True)

        if sum([args.train_only, args.test_only]) > 1:
            raise ValueError(
                'Cannot set --train_only true and --test_only true at the same time! One of them should be False.'
            )

        if args.train_agent is None:
            raise ValueError("--train_agent must be specified!")
        if args.train_only and args.train_epoch_num is None:
            raise ValueError("--train_epoch_num must be specified for training!")

        return args

    @classmethod
    def launch(cls):
        args = cls.configure()
        set_random_seeds(args.seed)

        device = torch.device("cuda:0")

        torch.set_float32_matmul_precision('medium')

        try:
            cls.worker_fn(args, device)
        except KeyboardInterrupt:
            print("Catch a KeyBoardInterrupt! Ready to exit the program...")

        sys.exit(0)

if __name__ == '__main__':
    Launcher.launch()