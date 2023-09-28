import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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
train_loader = tud.DataLoader(dataset_train, batch_size=64, num_workers=8)


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


syncopation_tokens = ["<", ">"]
syncopation_vector = torch.tensor(
    [0.0398, 0.5080, 0.4021, 0.2535, -0.2351, -0.0130, 0.6145, 0.2851]
)
lower_limit = torch.tensor(
    [-3.0488, -2.0577, -3.4109, -3.9441, -3.8085, -2.6445, -2.3709, -2.4634]
)
upper_limit = torch.tensor(
    [4.3636, 2.1375, 3.1498, 2.8253, 2.7260, 2.6107, 2.6847, 3.6353]
)


def find_limits():
    lower_limit = 1000 * torch.ones([model.latent_size])
    upper_limit = -1000 * torch.ones([model.latent_size])
    for x, y in train_loader:
        m, std = model.embed2dist(x)
        lower_limit = torch.minimum(lower_limit, torch.min(m, dim=0).values)
        upper_limit = torch.maximum(upper_limit, torch.max(m, dim=0).values)

    print(lower_limit)
    print(upper_limit)


@torch.no_grad()
def plot_change(attribute_vector, function, samples=256):
    # sample randomly
    vecs = lower_limit + (upper_limit - lower_limit) * torch.randn(
        samples, lower_limit.shape[0]
    )
    print("Sampling")
    # decode originals and shifted
    original = []
    res = []
    for v in vecs:
        sample = sample_z(v)
        original.append(sample)
        sample = sample_z(v + attribute_vector)
        res.append(sample)

    print("Stacking")
    # stack originals
    original = torch.stack(
        tuple(
            F.pad(
                torch.LongTensor(o),
                (0, 256 - len(o)),
                mode="constant",
                value=dataset.pad_idx,
            )
            for o in original
        ),
        dim=0,
    )
    # stack shifted
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
    print("Scoring")
    # compute scores
    o_scores = function(original)
    r_scores = function(res)

    print("Plotting")
    fig, axs = plt.subplots(vecs.shape[1])
    y = r_scores - o_scores
    for dim in range(vecs.shape[1]):
        x = vecs[:, dim]
        data = torch.stack((x, y)).detach().numpy()
        gm = GaussianMixture(n_components=2, random_state=0).fit(data)
        x = x.detach().numpy()
        X = np.linspace(min(x), max(x), num=100)
        Y = gm.predict_proba(X)
        print(Y)
        axs[dim].plot(X, Y)
    plt.show()


plot_change(syncopation_vector, contains_tokens(syncopation_tokens), samples=4)
