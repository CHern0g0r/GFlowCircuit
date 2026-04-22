"""TensorBoard logging helpers for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc, assignment]


class TensorBoardLogger:
    """Thin wrapper so training works without `tensorboard` installed."""

    def __init__(self, log_dir: Path | None) -> None:
        self._writer: Any = None
        if log_dir is None:
            return
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires the `tensorboard` package. "
                "Install with: pip install tensorboard"
            )
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalars(self, step: int, scalars: dict[str, float]) -> None:
        if self._writer is None:
            return
        for key, value in scalars.items():
            self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
