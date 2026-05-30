from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class BaseLogger(ABC):
    @abstractmethod
    def __enter__(self): ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback): ...

    @abstractmethod
    def log_msg(self, msg: str) -> None: ...

    @abstractmethod
    def log_scalar(
        self, name: str, value: float, step: Optional[int] = None
    ) -> None: ...


class SilentLogger(BaseLogger):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def log_msg(self, msg: str) -> None:
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        pass


class TensorboardLogger(BaseLogger):
    def __init__(self, save_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.save_dir = save_dir

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def log_msg(self, msg: str) -> None:
        print(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)

    def open(self) -> None:
        self.writer = SummaryWriter(log_dir=self.save_dir)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
