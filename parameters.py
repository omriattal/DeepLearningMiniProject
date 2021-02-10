import torch
FILE_NAMES = ["warandpeace"]
SEQ_LEN = 100
BATCH_SIZE = 100
SPLITS = (0, 80, 90, 100)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-3
EPOCHS = 50
CLIP = 5
PRECISION = 32
GAMMA = 0.95
LEARNING_RATE_DECAY = 10
