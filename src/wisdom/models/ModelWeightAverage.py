"""Evaluation-only exponential or stochastic averaging of trainable model state."""

from collections import OrderedDict

import torch
from torch import Tensor, nn

from wisdom.models.WeightAveragingMode import WeightAveragingMode


class ModelWeightAverage:
    """Maintain one averaged state while optimization continues on ordinary model weights."""

    def __init__(
        self,
        model          : nn.Module,
        mode           : WeightAveragingMode | str,
        ema_decay      : float = 0.99,
        swa_start_epoch: int = 1,
    ) -> None:
        """Copy initial model state and configure EMA or late-training arithmetic averaging.

        Args:
            model: Model whose complete floating state is averaged.
            mode: ``ema`` or ``swa``. ``none`` is rejected because no object is needed.
            ema_decay: EMA retention coefficient in ``[0,1)``.
            swa_start_epoch: First one-based epoch included in SWA.

        Raises:
            ValueError: If the mode or its numeric policy is invalid.
        """
        self.mode = WeightAveragingMode(mode)
        if self.mode is WeightAveragingMode.NONE:
            raise ValueError("no weight-average object is needed for mode='none'")
        if not 0.0 <= ema_decay < 1.0 or swa_start_epoch < 1:
            raise ValueError("weight averaging decay/start epoch is invalid")

        self.ema_decay       = float(ema_decay)
        self.swa_start_epoch = int(swa_start_epoch)
        self.average = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )
        self.backup : OrderedDict[str, Tensor] | None = None
        self.updates = 0

    @property
    def ready(self) -> bool:
        """Return whether at least one optimization update contributes to the average."""
        return self.updates > 0

    @torch.no_grad()
    def update(self, model: nn.Module, epoch: int) -> None:
        """Incorporate current model state after one optimizer step.

        Args:
            model: Optimized model with the same state keys used at construction.
            epoch: Current one-based training epoch.
        """
        if self.mode is WeightAveragingMode.SWA and epoch < self.swa_start_epoch:
            return
        state = model.state_dict()
        self.updates += 1
        for name, current in state.items():
            target = self.average[name]
            if not current.is_floating_point():
                target.copy_(current)
            elif self.mode is WeightAveragingMode.EMA:
                target.mul_(self.ema_decay).add_(current, alpha=1.0 - self.ema_decay)
            else:
                target.add_((current - target) / self.updates)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        """Swap averaged weights into the model for validation or checkpoint publication.

        Args:
            model: Model matching the averaged state.

        Raises:
            RuntimeError: If no update exists or an earlier swap has not been restored.
        """
        if not self.ready:
            raise RuntimeError("weight average is not ready")
        if self.backup is not None:
            raise RuntimeError("averaged weights are already applied")
        self.backup = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )
        model.load_state_dict(self.average)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore optimizer-owned weights after an averaged evaluation.

        Args:
            model: Model currently holding this object's averaged state.

        Raises:
            RuntimeError: If :meth:`apply` was not called first.
        """
        if self.backup is None:
            raise RuntimeError("averaged weights are not applied")
        model.load_state_dict(self.backup)
        self.backup = None
