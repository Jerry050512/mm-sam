import argparse
import logging
import sys
import os
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

from typing import Dict, Union, Callable, Iterator, Sequence, Mapping
from torch.cuda.amp import autocast

from utilbox.log_utils import dict_to_log_message
from utilbox.train_agents.base import TrainAgent
from utilbox.import_utils import import_class


def batch_to_cuda(batch: Dict, device: Union[str, torch.device], non_blocking: bool = False) -> Dict:
    if isinstance(device, str):
        device = torch.device(device)

    def data_to_cuda(input_data):
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
            return input_data.to(device=device, dtype=input_data.dtype, non_blocking=non_blocking)
        elif isinstance(input_data, torch.Tensor):
            return input_data.to(device=device, dtype=input_data.dtype, non_blocking=non_blocking)
        elif isinstance(input_data, str) or input_data is None:
            return input_data
        elif isinstance(input_data, Sequence):
            return [data_to_cuda(item) for item in input_data]
        elif isinstance(input_data, Mapping):
            return {data_key: data_to_cuda(data_value) for data_key, data_value in input_data.items()}
        elif isinstance(input_data, int):
            return torch.LongTensor([input_data]).to(device=device, non_blocking=non_blocking)
        elif isinstance(input_data, float):
            return torch.FloatTensor([input_data]).to(device=device, non_blocking=non_blocking)
        elif isinstance(input_data, bool):
            return torch.BoolTensor([input_data]).to(device=device, non_blocking=non_blocking)
        else:
            raise TypeError(f"Unsupported data type: {type(input_data)}!")

    return data_to_cuda(batch)


class TrainManager:

    def __init__(self, device: torch.device, args: argparse.Namespace):
        self.device = device
        self.args = args

        log_dir = '/hy-tmp/output'
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'result.log')

        if not args.test_only and os.path.exists(self.log_file):
             with open(self.log_file, 'w') as f:
                 f.write("")

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            filename=self.log_file,
                            filemode='a')
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.use_amp = self.args.use_amp
        self.train_epoch_num = self.args.train_epoch_num
        assert hasattr(args, 'train_agent')
        self.train_agent: TrainAgent = import_class(args.train_agent)(
            seed=args.seed, device=self.device,
            train_only=args.train_only, test_only=args.test_only,
            use_amp=self.use_amp,
            train_bs=args.train_bs, train_workers_per_bs=args.train_workers_per_bs,
            train_epoch_num=self.train_epoch_num, pin_memory=True,
            valid_bs=args.valid_bs, valid_workers_per_bs=args.valid_workers_per_bs,
            test_bs=args.test_bs, test_workers_per_bs=args.test_workers_per_bs,
            folder_path=args.launch_folder_path, **args.agent_kwargs
        )
        if not isinstance(self.train_agent, TrainAgent):
            raise TypeError("Your training agent must be an instance of utilbox.train_agents.base.TrainAgent!")
        self.train_batch_num = self.train_agent.train_batch_num
        self.valid_batch_num = self.train_agent.valid_batch_num
        self.test_batch_num = self.train_agent.test_batch_num

    def batch_preprocess(self, batch: Dict):
        return batch_to_cuda(batch, self.device, non_blocking=True)

    def save_checkpoint(self, saved_ckpt_path):
        if not saved_ckpt_path.endswith('.pth'):
            saved_ckpt_path += '.pth'
        saved_ckpt_path = saved_ckpt_path.replace(' ', '_')

        ckpt_pardir = os.path.dirname(saved_ckpt_path)
        os.makedirs(ckpt_pardir, exist_ok=True)
        torch.save(self.train_agent.state_dict()["model"], saved_ckpt_path)

    def train(self):
        self.logger.info("Start training!")
        self.train_agent.train()
        for epoch in range(1, self.train_epoch_num + 1):
            self.logger.info(f"==> Starting epoch {epoch}/{self.train_epoch_num}")

            pbar = tqdm(total=self.train_batch_num, desc=f'Epoch {epoch}/{self.train_epoch_num}', file=sys.stdout)

            self.train_agent.before_train_epoch()
            train_iter = iter(self.train_agent.train_loader)
            for step in range(1, self.train_batch_num + 1):
                total_step_num = (epoch - 1) * self.train_batch_num + step
                batch = next(train_iter)
                if not isinstance(batch, Dict):
                    raise TypeError(
                        f"collate_fn() of your Dataset should return a Dict, but got {type(batch)}!"
                    )
                batch = self.batch_preprocess(batch)

                with autocast(enabled=self.use_amp):
                    train_losses = self.train_agent.train_step(batch, epoch=epoch, step=step)

                if not isinstance(train_losses, Dict):
                    raise TypeError(
                        "train_step() of your model should return a Dict of trainable torch.Tensor, "
                        f"but got {type(train_losses)}!"
                    )
                for key, value in train_losses.items():
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"The {key} loss from your train_step() should be a torch.Tensor, "
                            f"but got {type(value)}!"
                        )

                self.train_agent.before_optim_step()
                self.train_agent.optim_step(train_losses, total_step_num=total_step_num)
                self.train_agent.after_optim_step()

                train_losses_detach = {key: value.clone().detach() for key, value in train_losses.items()}

                log_msg = "lr: {:.2e}".format(self.train_agent.get_optim_lr()["main_opt"])
                for l_name, l_value in train_losses_detach.items():
                    log_msg += " {}: {:.4f}".format(l_name, l_value.item())

                pbar.set_postfix_str(log_msg)
                pbar.update(1)

                if step % 100 == 0:
                    self.logger.info(f"Epoch {epoch}/{self.train_epoch_num}, Step {step}/{self.train_batch_num}: {log_msg}")

            pbar.close()

            if self.train_agent.valid_loader is not None and self.valid_batch_num > 0:
                self.train_agent.before_valid_epoch()
                self.train_agent.eval()
                valid_results = self.eval(
                    eval_iterator=iter(self.train_agent.valid_loader),
                    total_eval_batch_num=self.valid_batch_num,
                    eval_step_fn=self.train_agent.valid_step,
                    eval_result_fn=self.train_agent.get_valid_results
                )
                self.train_agent.after_valid_epoch()
                self.logger.info(f"Validation Results: {dict_to_log_message(valid_results)}")
                self.train_agent.train()

            self.train_agent.after_train_epoch()

            checkpoint_path = '/hy-tmp/output/checkpoint.pth'
            self.save_checkpoint(checkpoint_path)
            self.logger.info(f"Epoch {epoch} checkpoint saved to {checkpoint_path}")

        self.logger.info("Finish training!")
        torch.cuda.empty_cache()

    def eval(
            self, eval_iterator: Iterator,
            total_eval_batch_num: int,
            eval_step_fn: Callable, eval_result_fn: Callable, desc="Evaluating"
    ):
        with torch.no_grad():
            pbar = tqdm(total=total_eval_batch_num, desc=desc, file=sys.stdout)
            for step in range(1, total_eval_batch_num + 1):
                batch = next(eval_iterator)
                batch = self.batch_preprocess(batch)
                eval_step_fn(batch)
                pbar.update(1)
            pbar.close()
        return eval_result_fn()

    def test(self):
        self.logger.info("Start testing!")

        ckpt_path = self.args.ckpt_path
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.logger.info(f'Start testing on the checkpoint:{ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.train_agent.load_model_state_dict(ckpt)
        else:
            self.logger.warning(f'No checkpoint found at {ckpt_path}! Please be careful!')

        self.train_agent.before_test()
        self.train_agent.eval()

        test_results = self.eval(
            eval_iterator=iter(self.train_agent.test_loader),
            total_eval_batch_num=self.test_batch_num,
            eval_step_fn=self.train_agent.test_step,
            eval_result_fn=self.train_agent.get_test_results,
            desc="Testing"
        )
        self.train_agent.after_test()

        self.logger.info(f"Test Results: {dict_to_log_message(test_results)}")
        self.logger.info("Finish testing!")