import argparse
import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from model_v2 import *
from dataset import *


# def main(args):
torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)

# load dataset
dataset = FolkDataset(
    "./data/dataset.txt",
    max_sequence_length=256,
)

dataset_train, dataset_val = tud.random_split(dataset, [0.95, 0.05])

batch_size = 64
train_loader = tud.DataLoader(dataset_train, batch_size=batch_size, num_workers=8)
val_loader = tud.DataLoader(dataset_val, batch_size=batch_size, num_workers=8)

"""
# load model
model = FolktuneVAE(
    input_size=dataset.vocab_size,
    embedding_size=64,
    hidden_size=64,
    num_layers=2,
    latent_size=8,
    output_size=dataset.vocab_size,
    dropout=0.5,
    pad_index=dataset.pad_idx,
)
"""
model = FolktuneVAE.load_from_checkpoint(
    "tb_logs/conductor-folktuneVAE/version_0/checkpoints/epoch=49-step=15750.ckpt"
)

# logger
logger = pll.TensorBoardLogger("tb_logs", name="conductor-folktuneVAE")

# training
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    logger=logger,
    callbacks=[
        # Early stopping
        plc.EarlyStopping(
            monitor="val/val_loss",
            mode="min",
            patience=3,
            verbose=True,
            strict=True,
            min_delta=1e-4,
        ),
        # saves top-K checkpoints based on "val_loss" metric
        plc.ModelCheckpoint(
            save_top_k=1,
            monitor="val/val_loss",
            mode="min",
        ),
    ],
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=16)
    parser.add_argument("-es", "--embedding_size", type=int, default=16)
    parser.add_argument("-ls", "--latent_size", type=int, default=16)
    parser.add_argument("-l", "--layers", type=int, default=2)
    parser.add_argument("-b", "--bidirectional", action="store_true")
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-bs", "--batch_size", type=int, default=100)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.9)
    parser.add_argument("-p", "--patience", type=int, default=3)
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=256)
    parser.add_argument("-en", "--experiment_name", type=str, default="")
    args = parser.parse_args()

    main(args)
"""
