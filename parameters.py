import torch
FILE_NAMES = ["sherlock","frankenstein"]
SEQ_LEN = 100
BATCH_SIZE = 100
SPLITS = (0, 80, 90, 100)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-3
EPOCHS = 50
CLIP = 5
DROPOUT = 0.0
PRECISION = 32