import torch
FILE_NAMES = ["sherlock"]
SEQ_LEN = 100
BATCH_SIZE = 100
SPLITS = (0, 80, 90, 100)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.01
EPOCHS = 50
CLIP = 5
PRECISION = 32
TRUNCATED_BPTT_STEPS = 10
