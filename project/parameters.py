import torch

TIME_STEPS = 100
BATCH_SIZE = 100
SPLITS = (0, 80, 10, 10)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
