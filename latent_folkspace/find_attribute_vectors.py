import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from model_v2 import *
from dataset import *

# load model
checkpoint = "latest.ckpt"
model = FolktuneVAE.load_from_checkpoint(checkpoint)
model.eval()

torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)
random.seed(0)

# load dataset
dataset = FolkDataset(data_file="data/dataset.txt", max_sequence_length=256)
# get train data
dataset_train, dataset_test = tud.random_split(dataset, [0.95, 0.05])

# vocabs
i2w = dataset.i2w
w2i = dataset.w2i

# choose N random train examples
train_loader = tud.DataLoader(dataset_train, batch_size=1024, num_workers=8)

"""
radius = 0
# find boundaries
for x, y in train_loader:
    z = model.embed(x)
    radius = float(max(radius, max(torch.norm(z, p=2, dim=1))))
print(radius)
"""

radius = 5.623257160186768

N = len(dataset_train)
train_loader = tud.DataLoader(dataset_train, batch_size=N, num_workers=8)
data, data_y = next(iter(train_loader))
data = data.detach()


def find_vector(dataset, f):
    # assign attribute score to data
    score = f(dataset)

    # sort
    sorted_indexes = np.argsort(score)

    # get per-attribute quartiles
    quarter = len(score) // 4

    # lower
    lw_indexes = sorted_indexes[:quarter]
    lower_quartile = dataset[lw_indexes]

    # higher
    up_indexes = sorted_indexes[-quarter:]
    upper_quartile = dataset[up_indexes]

    # embed quartiles & average
    lower_z = model.embed(lower_quartile)
    lower_vector = torch.mean(lower_z, dim=0)

    upper_z = model.embed(upper_quartile)
    upper_vector = torch.mean(upper_z, dim=0)

    vector = upper_vector - lower_vector

    return F.normalize(vector, dim=0).detach()
    # return vector.detach().numpy()


# find vectors
def contains_tokens(tokens):
    NONE = -1

    def f(in_data):
        data = in_data.clone()

        padded = data.clone()
        padded[padded == w2i["<pad>"]] = NONE
        padded[padded != NONE] = 1
        padded[padded == NONE] = 0
        padded = padded.sum(dim=-1)

        for tok in tokens:
            data[data == w2i[tok]] = NONE

        data[data != NONE] = 0
        data[data == NONE] = 1
        counts = torch.div(data.sum(dim=-1), padded)

        return counts

    return f


def length(data):
    NONE = -1
    tokens = data.clone()
    tokens[tokens != w2i["<pad>"]] = NONE
    tokens[tokens != NONE] = 0
    tokens[tokens == NONE] = 1
    tokens = tokens.sum(dim=-1)

    return tokens


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

# token_list = ["C,", "C", "c", "c'", "=C,", "=C", "=c", "=c'"]
token_list = ["<", ">"]
# token_list = [w for w in w2i if ("^" in w or "_" in w or "=" in w)]
# token_list = [w for w in w2i if "^" in w]
# token_list = ["K:Cmix"]
# print(token_list)
vec = find_vector(data, length)
print(vec)


def sample_z(z):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    sample = model.inference(
        dataset.sos_idx,
        dataset.eos_idx,
        max_len=256,
        mode="greedy",
        temperature=1,
        PK=0.95,
        prompt=None,
        z=z,
    )
    return sample


"""
# sample random z
# z = -radius * torch.ones([1, len(vec)])
z = -radius * vec.unsqueeze(0)
print(z.shape)
sample = sample_z(z)
print_tune(sample)

# more C!
steps = 10
z1 = z
res = [sample]
for i in range(steps):
    z1 += vec
    sample = sample_z(z1)
    res.append(sample)
    print_tune(sample)

res = torch.stack(
    tuple(
        F.pad(
            torch.LongTensor(r),
            (0, 256 - len(r)),
            mode="constant",
            value=dataset.pad_idx,
        )
        for r in res
    ),
    dim=0,
)
print(length(res))
"""
