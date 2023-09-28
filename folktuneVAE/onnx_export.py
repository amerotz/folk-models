import torch
import random

from model import *
from dataset import *

torch.manual_seed(0)
random.seed(0)

# load dataset
dataset = FolkDataset("./data/dataset.txt", max_sequence_length=256)

checkpoint = "tb_logs/folktuneVAE/version_0/checkpoints/epoch=28-step=17284.ckpt"
model = FolktuneVAE.load_from_checkpoint(checkpoint).to("cpu")

dummy_input = (
    torch.LongTensor([dataset.sos_idx] + [dataset.pad_idx] * 255).unsqueeze(0).to("cpu")
)
print(dummy_input.shape)

input_names = ["input"]
output_names = ["output", "mean", "logv"]

torch.onnx.export(
    model,
    dummy_input,
    "folktuneVAE.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
)

obj = {
    "w2i": dataset.w2i,
    "i2w": dataset.i2w,
    "sos": dataset.sos_idx,
    "eos": dataset.eos_idx,
    "pad": dataset.pad_idx,
    "primer": dataset.pad_idx,
    "checkpoint": "./models/folktuneVAE.onnx",
}
with open("folktuneVAE.json", "w") as f:
    json.dump(obj, f)
