BRAIN-CP full training ready snapshot
created_at=20260609_150340

Use trainer:
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerV2_BRAINCP

Important:
- BRAIN-CP ON for training dataloader
- BRAIN-CP OFF for validation dataloader
- _final trainer name is no longer used
- torch.compile OFF
