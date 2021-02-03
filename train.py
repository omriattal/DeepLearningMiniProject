import pytorch_lightning as pl
from data import dataloader
from parameters import *
from net.network import Network
from net.mylightningmodule import MyLightningModule
import os
for file_name in FILE_NAMES:
    file_path = f"data/{file_name}.txt"
    (train, test, val), vocabulary = dataloader.load_data(file_path, SPLITS, BATCH_SIZE, SEQ_LEN, True, DEVICE)
    for model_name in ["lstm", "gru", "rnn"]:
        for num_layers in [1, 2]:
        # for num_layers in [1, 2, 3]:
            for hidden_size in [32]:
            # for hidden_size in [32, 64, 128, 256, 512]:
                network = Network(len(vocabulary), hidden_size, model_name, DROPOUT, num_layers, DEVICE)
                my_lightning = MyLightningModule(network, LR)
                # TODO: early stopping?
                network_trainer = pl.Trainer(gpus=int(DEVICE == "cuda"), precision=PRECISION, gradient_clip_val=CLIP,
                                             max_epochs=EPOCHS, progress_bar_refresh_rate=10,
                                             benchmark=True)
                network_trainer.fit(my_lightning, train_dataloader=train, val_dataloaders=val)
                network.save_network()
