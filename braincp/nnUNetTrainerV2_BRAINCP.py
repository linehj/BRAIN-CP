"""
nnUNetTrainerV2_BRAINCP.py

Clean BRAIN-CP trainer wrapper for nnU-Net v2.

Important:
BRAIN-CP augmentation is applied inside the nnU-Net dataloader:

    nnunetv2/training/dataloading/data_loader.py

This trainer intentionally does not override train_step().
Therefore, full training uses the standard nnU-Net v2 training loop,
including the default loss, optimizer, AMP/GradScaler behavior,
checkpointing, validation, and logging.

Optional:
Set BRAINCP_NUM_EPOCHS only for smoke tests.
For full training, leave BRAINCP_NUM_EPOCHS unset.
"""

from __future__ import annotations

import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerV2_BRAINCP(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device=None,
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        num_epochs = os.environ.get("BRAINCP_NUM_EPOCHS")
        if num_epochs is not None and num_epochs != "":
            self.num_epochs = int(num_epochs)

        num_iterations = os.environ.get("BRAINCP_NUM_ITERATIONS_PER_EPOCH")
        if num_iterations is not None and num_iterations != "":
            self.num_iterations_per_epoch = int(num_iterations)

        num_val_iterations = os.environ.get("BRAINCP_NUM_VAL_ITERATIONS_PER_EPOCH")
        if num_val_iterations is not None and num_val_iterations != "":
            self.num_val_iterations_per_epoch = int(num_val_iterations)

    @staticmethod
    def _set_braincp_enabled_recursive(obj, enabled: bool, _seen=None, _depth: int = 0) -> int:
        if obj is None or _depth > 8:
            return 0
        if _seen is None:
            _seen = set()
        oid = id(obj)
        if oid in _seen:
            return 0
        _seen.add(oid)

        changed = 0
        if hasattr(obj, "braincp_enabled"):
            try:
                setattr(obj, "braincp_enabled", bool(enabled))
                changed += 1
            except Exception:
                pass

        children = []
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            children = list(obj)
        else:
            try:
                children = list(vars(obj).values())
            except Exception:
                children = []

        for child in children:
            if isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            changed += nnUNetTrainerV2_BRAINCP._set_braincp_enabled_recursive(
                child, enabled, _seen, _depth + 1
            )
        return changed

    def get_dataloaders(self):
        dl_tr, dl_val = super().get_dataloaders()
        if os.environ.get("BRAINCP_AUGMENT_VAL", "0") != "1":
            changed = self._set_braincp_enabled_recursive(dl_val, False)
            self.print_to_log_file(
                f"[BRAIN-CP] validation dataloader augmentation disabled. objects_changed={changed}"
            )
            if changed <= 0:
                raise RuntimeError(
                    "BRAIN-CP validation disable failed: no validation dataloader object was changed."
                )
        return dl_tr, dl_val

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            "[BRAIN-CP] trainer initialized. "
            "BRAIN-CP augmentation is applied in the dataloader before image-level transforms."
        )
