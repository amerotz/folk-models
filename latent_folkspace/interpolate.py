import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from model_v2 import *
from dataset import *
import random

torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)
random.seed(0)

# load dataset
dataset = FolkDataset(data_file="data/dataset.txt", max_sequence_length=256)
# get train data
dataset_train, dataset_test = tud.random_split(dataset, [0.95, 0.05])
w2i = dataset.w2i
i2w = dataset.i2w

# load model
checkpoint = "latest.ckpt"
model = FolktuneVAE.load_from_checkpoint(checkpoint)
model.eval()


def sample_z(z):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    sample = model.inference(
        dataset.sos_idx,
        dataset.eos_idx,
        max_len=256,
        mode="topp",
        temperature=1,
        PK=0.95,
        prompt=None,
        z=z,
    )
    return sample


def print_tune(gen):
    tune = [i2w[str(g)] for g in gen[1:-1]]
    s = f"X:{print_tune.num}\n{tune[0]}\nL:1/8\n{tune[1]}\n"
    for t in tune[2:]:
        s += t
        if t == ":|":
            s += "\n"
    s += "\n"
    print_tune.num += 1
    print(s)


print_tune.num = 0

z1 = torch.randn([1, model.latent_size])
z2 = z1 + 0.1 * torch.ones([1, model.latent_size])

steps = 10
for i in range(steps + 1):
    perc = i / steps
    z = z2 * perc + z1 * (1 - perc)
    tune = sample_z(z)
    print_tune(tune)
