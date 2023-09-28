import random
import argparse
import umap
import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from model_v2 import *
from dataset import *


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(0)
    random.seed(0)

    # load dataset
    dataset = FolkDataset("./data/dataset.txt", max_sequence_length=256)

    dataset_train, dataset_val = tud.random_split(
        dataset, [args.train_ratio, 1 - args.train_ratio]
    )

    # load model
    model = FolktuneVAE.load_from_checkpoint(args.checkpoint)
    model.eval()

    keys = defaultdict(list)
    times = defaultdict(list)
    both = defaultdict(list)
    all_z = []
    for x, y in dataset_val:
        k = dataset.i2w[str(x[2].item())]
        t = dataset.i2w[str(x[1].item())]

        x = x.unsqueeze(0)
        x = x.to(model.device)
        z = model.embed(x).squeeze(0).detach().cpu().numpy()

        all_z.append(z)

        keys[k].append(z)
        times[t].append(z)
        both[f"{k}_{t}"].append(z)

    scaled_data = StandardScaler().fit_transform(all_z)

    # reducer = umap.UMAP()
    reducer = PCA(n_components=2, svd_solver="full")
    """
    reducer = TSNE(n_components=2)
    """

    reducer.fit(scaled_data)

    data = keys
    palette = sns.color_palette("Spectral", len(data))
    for i, k in enumerate(data):
        embedding = reducer.transform(data[k])
        plt.scatter(embedding[:, 0], embedding[:, 1], label=k, color=palette[i])

    plt.legend()
    plt.gca().set_aspect("equal", "datalim")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.9)
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=256)
    args = parser.parse_args()

    main(args)
