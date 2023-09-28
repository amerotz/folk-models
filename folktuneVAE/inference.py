import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from model_v2 import *
from dataset import *


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(0)
    random.seed(0)

    # load dataset
    dataset = FolkDataset("./data/dataset.txt", max_sequence_length=256)

    # load model
    model = FolktuneVAE.load_from_checkpoint(args.checkpoint)
    model.eval()

    s = ""
    z = None

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    for i in range(args.num_samples):
        prompt = None
        if args.prompt != None:
            prompt = [dataset.sos_idx] + [
                dataset.w2i[w] for w in args.prompt.split(" ")
            ]
            x = torch.LongTensor(prompt)
            x = x.unsqueeze(0)
            x = x.to(model.device)
            z = model.embed(prompt)

        gen = model.inference(
            dataset.sos_idx,
            dataset.eos_idx,
            max_len=args.max_sequence_length,
            mode=args.mode,
            temperature=args.temperature,
            PK=args.pk,
            prompt=prompt,
            z=z,
        )

        tune = [dataset.i2w[str(g)] for g in gen[1:-1]]
        s += f"{tune[0]}\nL:1/8\n{tune[1]}\n"
        for t in tune[2:]:
            s += t
            if t == ":|":
                s += "\n"
        s += "\n\n"

    print(s)

    with open(f"results_{args.mode}.txt", "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str)
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-m", "--mode", type=str, default="topp")
    parser.add_argument("-pk", "--pk", type=float, default=1.0)
    parser.add_argument("-n", "--num_samples", type=int, default=100)
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=256)
    parser.add_argument("-p", "--prompt", type=str, default=None)
    parser.add_argument("-r", "--reconstruct", action="store_true", default=False)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
