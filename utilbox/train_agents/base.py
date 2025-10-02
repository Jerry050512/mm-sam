from typing import Dict, Union, Iterator, Optional, Protocol, TypeVar, runtime_checkable

import torch
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from utilbox.data_load.loader_utils import build_dataloader
from utilbox.optim_agents.base import OptimAgent


T = TypeVar('T')


@runtime_checkable
class IterableAndSized(Protocol[T]):
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...


class TrainAgent(ABC):
    """Abstract base class for all training agents.

    To implement your own training agent, inherit from this class and implement all the abstract methods:
        - agent_init()
        - train_step()
        - valid_step()
        - get_valid_results()
        - test_step()
        - get_test_results()

    This class has some attribute methods compatible with torch.nn.Module:
        - train()
        - eval()
        - parameters()
        - named_parameters()
        - model_state_dict() -> state_dict() of torch.nn.Module
        - load_model_state_dict() -> load_state_dict() of torch.nn.Module

    """
    def __init__(self,
                 # environment
                 device: Union[str, torch.device] = 'cuda',
                 seed: int = 3407,
                 train_only: bool = False,
                 test_only: bool = False,
                 use_amp: bool = False,
                 train_epoch_num: int = 0,
                 folder_path: str = None,
                 # data loading
                 train_bs: int = 1,
                 train_workers_per_bs: float = 0.5,
                 valid_bs: int = 1,
                 valid_workers_per_bs: float = 0.5,
                 test_bs: int = 1,
                 test_workers_per_bs: float = 0.5,
                 pin_memory: bool = False,
                 # customized arguments including optimization and agent arguments
                 **init_kwargs):

        if sum([train_only, test_only]) > 1:
            raise ValueError('Cannot set train_only and test_only to True at the same time! '
                             'Only one one of them should be true.')

        # register the environment variables
        self.seed = seed
        self.device = device
        self.train_only = train_only
        self.test_only = test_only
        self.folder_path = folder_path

        # register the arguments from the TrainManager for your reference in agent_init()
        if not self.test_only:
            assert train_bs > 0, "`train_bs` must be greater than 0."
            assert valid_bs > 0, "`valid_bs` must be greater than 0."
        if not self.train_only:
            assert test_bs > 0, "`test_bs` must be greater than 0."
        self._manager_config = dict(
            train_bs=train_bs,
            train_workers_per_bs=train_workers_per_bs,
            valid_bs=valid_bs,
            valid_workers_per_bs=valid_workers_per_bs,
            test_bs=test_bs,
            test_workers_per_bs=test_workers_per_bs,
            pin_memory=pin_memory,
            use_amp=use_amp,
            train_epoch_num=train_epoch_num,
        )

        # register protected attributes before agent_init()
        self._train_dataset, self._valid_dataset, self._test_dataset = None, None, None
        self._train_loader, self._valid_loader, self._test_loader = None, None, None
        self._optim_agent = None

        # Hook: something to be done before customized initialization of the specific agents
        self.before_agent_init(**init_kwargs)
        all_init_kwargs = {**self._manager_config, **init_kwargs}
        self.agent_init(**all_init_kwargs)
        # Hook: something to be done after customized initialization of the specific agents
        self.after_agent_init(**init_kwargs)

        # register pointers of Module attributes into protected Dict attribute for reference
        self._model_dict: Dict[str, torch.nn.Module] = {}
        attr_name_list = list(self.__dict__.keys())
        for name in attr_name_list:
            attr = getattr(self, name)
            if isinstance(attr, torch.nn.Parameter):
                raise RuntimeError(
                    "Please do not register torch.nn.Parameter in your TrainAgent since it won't be recorded! "
                    "Instead, it would be better if you wrap it as a torch.nn.Module."
                )
            if not isinstance(attr, torch.nn.Module):
                continue

            model = attr.to(device=self.device)
            setattr(self, name, model)
            self._model_dict[name] = model

    @abstractmethod
    def agent_init(self, **init_kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Dict, epoch: int, step: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch: Dict, iter_name: str = None):
        raise NotImplementedError

    @abstractmethod
    def get_valid_results(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: Dict, iter_name: str = None):
        raise NotImplementedError

    @abstractmethod
    def get_test_results(self) -> (Dict, Dict):
        raise NotImplementedError

    def infer(self, batch: Dict) -> Dict:
        raise NotImplementedError

    def train(self):
        for model in self._model_dict.values():
            model.train()

    def eval(self):
        for model in self._model_dict.values():
            model.eval()

    def parameters(self):
        para_ptr_list = []
        for model in self._model_dict.values():
            for para in model.parameters():
                if para.data_ptr() not in para_ptr_list:
                    para_ptr_list.append(para.data_ptr())
                    yield para

    def named_parameters(self):
        para_ptr_list = []
        for name, model in self._model_dict.items():
            for key, para in model.named_parameters():
                if para.data_ptr() not in para_ptr_list:
                    para_ptr_list.append(para.data_ptr())
                    yield f"{name}.{key}", para

    def model_state_dict(self) -> Dict:
        model_states = {}
        if len(self._model_dict) == 1:
            model_name = list(self._model_dict.keys())[0]
            model = self._model_dict[model_name]
            model_states = model.state_dict()
        else:
            for name, model in self._model_dict.items():
                model_states.update({f'{name}.{key}': value for key, value in model.state_dict().items()})
        return model_states

    def load_model_state_dict(self, model_state_dict: Dict):
        if len(self._model_dict) == 1:
            model_name = list(self._model_dict.keys())[0]
            model = self._model_dict[model_name]
            model.load_state_dict(model_state_dict)
        else:
            for name, model in self._model_dict.items():
                sub_model_states = {key.lstrip(f'{name}.'): value for key, value in model_state_dict.items()
                                    if key.startswith(f'{name}.')}
                assert len(sub_model_states) > 0, f"No sub models named {name} found in model_state_dict"
                model.load_state_dict(sub_model_states)

    def state_dict(self) -> Dict:
        return dict(optim_agent=self.optim_agent.state_dict(), model=self.model_state_dict())

    def load_state_dict(self, state_dict: Dict):
        assert 'optim_agent' in state_dict, "Your given state_dict must contain 'optim_agent'!"
        self.optim_agent.load_state_dict(state_dict['optim_agent'])
        assert 'model' in state_dict, "Your given state_dict must contain 'model_dict'!"
        self.load_model_state_dict(state_dict['model'])

    def optim_step(self, train_losses: Dict[str, torch.Tensor], total_step_num: int):
        self.optim_agent.step(losses=train_losses, total_step_num=total_step_num)

    def get_optim_lr(self) -> Dict[str, float]:
        return self.optim_agent.get_optim_lr()

    @property
    def optim_agent(self) -> OptimAgent:
        assert self._optim_agent is not None, "Please register your optimization agent in agent_init() method!"
        return self._optim_agent

    @optim_agent.setter
    def optim_agent(self, value: OptimAgent):
        assert isinstance(value, OptimAgent), (
            "Your registered optimization agent must be an instance of utilbox.optim_agents.base.OptimAgent!"
        )
        self._optim_agent = value

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        assert self._train_dataset is not None, (
            "Please register your training Dataset in agent_init() method by calling `self.train_dataset = ...`!"
        )
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value: torch.utils.data.Dataset):
        assert isinstance(value, torch.utils.data.Dataset), (
            "Your registered training dataset must be an instance of torch.utils.data.Dataset!"
        )
        self._train_dataset = value
        self.train_loader = build_dataloader(
            dataset=self._train_dataset,
            worker_num=round(self.manager_config["train_bs"] * self.manager_config["train_workers_per_bs"]),
            batch_size=self.manager_config["train_bs"],
            shuffle=True,
            drop_last=True,
            pin_memory=self.manager_config["pin_memory"],
            distributed=False,
            worker_seed=self.seed
        )

    @property
    def train_loader(self) -> Optional[IterableAndSized]:
        return self._train_loader

    @train_loader.setter
    def train_loader(self, value: IterableAndSized):
        assert isinstance(value, IterableAndSized), "Your registered training dataloader must be IterableAndSized!"
        self._train_loader = value

    @property
    def valid_dataset(self) -> Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]:
        return self._valid_dataset

    @valid_dataset.setter
    def valid_dataset(self, value: Union[torch.utils.data.Dataset, Dict]):
        assert isinstance(value, (torch.utils.data.Dataset, Dict)), (
            "Your registered validation dataset must be an instance of torch.utils.data.Dataset or "
            "a Dict of torch.utils.data.Dataset instances!"
        )
        self._valid_dataset = value

        valid_worker_num = round(self.manager_config["valid_bs"] * self.manager_config["valid_workers_per_bs"])
        if isinstance(self._valid_dataset, torch.utils.data.Dataset):
            self.valid_loader = build_dataloader(
                dataset=self._valid_dataset,
                worker_num=valid_worker_num,
                batch_size=self.manager_config["valid_bs"],
                shuffle=False,
                drop_last=False,
                pin_memory=self.manager_config["pin_memory"],
                distributed=False,
                worker_seed=self.seed
            )
        else:
            self.valid_loader = {
                v_name: build_dataloader(
                    dataset=v_dataset,
                    worker_num=valid_worker_num,
                    batch_size=self.manager_config["valid_bs"],
                    shuffle=False,
                    drop_last=False,
                    pin_memory=self.manager_config["pin_memory"],
                    distributed=False,
                    worker_seed=self.seed
                ) for v_name, v_dataset in self._valid_dataset.items()
            }

    @property
    def valid_loader(self) -> Union[IterableAndSized, Dict[str, IterableAndSized], None]:
        return self._valid_loader

    @valid_loader.setter
    def valid_loader(self, value: Union[IterableAndSized, Dict[str, IterableAndSized]]):
        assert isinstance(value, (IterableAndSized, Dict)), (
            "Your registered validation dataloader must be an instance of IterableAndSized "
            "or a Dict of IterableAndSized instances!"
        )
        self._valid_loader = value

    @property
    def test_dataset(self) -> Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]:
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value: Union[torch.utils.data.Dataset, Dict]):
        assert isinstance(value, (torch.utils.data.Dataset, Dict)), (
            "Your registered testing dataset must be an instance of torch.utils.data.Dataset or "
            "a Dict of torch.utils.data.Dataset instances!"
        )
        self._test_dataset = value

        test_worker_num = round(self.manager_config["test_bs"] * self.manager_config["test_workers_per_bs"])
        if isinstance(self._test_dataset, torch.utils.data.Dataset):
            self.test_loader = build_dataloader(
                dataset=self._test_dataset,
                worker_num=test_worker_num,
                batch_size=self.manager_config["test_bs"],
                shuffle=False,
                drop_last=False,
                pin_memory=self.manager_config["pin_memory"],
                distributed=False,
                worker_seed=self.seed
            )
        else:
            self.test_loader = {
                t_name: build_dataloader(
                    dataset=t_dataset,
                    worker_num=test_worker_num,
                    batch_size=self.manager_config["test_bs"],
                    shuffle=False,
                    drop_last=False,
                    pin_memory=self.manager_config["pin_memory"],
                    distributed=False,
                    worker_seed=self.seed
                ) for t_name, t_dataset in self._test_dataset.items()
            }

    @property
    def test_loader(self) -> Union[IterableAndSized, Dict[str, IterableAndSized]]:
        return self._test_loader

    @test_loader.setter
    def test_loader(self, value: Union[IterableAndSized, Dict[str, IterableAndSized]]):
        assert isinstance(value, (IterableAndSized, Dict)), (
            "Your registered testing dataloader must be an instance of IterableAndSized "
            "or a Dict of IterableAndSized instances!"
        )
        self._test_loader = value

    @property
    def train_batch_num(self) -> int:
        if self.train_loader is None:
            return 0
        return len(self.train_loader)

    @property
    def valid_batch_num(self) -> Union[int, Dict[str, int]]:
        if self.valid_loader is None:
            return 0
        if isinstance(self.valid_loader, Dict):
            return {v_name: len(v_dataloader) for v_name, v_dataloader in self.valid_loader.items()}
        else:
            return len(self.valid_loader)

    @property
    def test_batch_num(self) -> Union[int, Dict[str, int]]:
        if isinstance(self.test_loader, Dict):
            return {t_name: len(t_dataloader) for t_name, t_dataloader in self.test_loader.items()}
        else:
            return len(self.test_loader)

    @property
    def manager_config(self):
        return self._manager_config

    def before_agent_init(self, **init_kwargs):
        pass

    def after_agent_init(self, **init_kwargs):
        pass

    def before_train_epoch(self):
        pass

    def before_optim_step(self):
        pass

    def after_optim_step(self):
        pass

    def after_train_epoch(self):
        pass

    def before_valid_epoch(self):
        pass

    def after_valid_epoch(self):
        pass

    def before_test(self):
        pass

    def after_test(self):
        pass