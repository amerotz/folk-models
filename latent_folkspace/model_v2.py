import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np
from torch.nn import functional as F


class FolktuneVAE(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        latent_size: int,
        output_size: int,
        dropout: float,
        # bidirectional: bool,
        pad_index: int,
    ):
        super().__init__()

        self.t = 0
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=pad_index)
        self.encoder = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            # bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        # self.multiplier = 2 if bidirectional else 1
        self.multiplier = 1

        self.hidden2mean = nn.Linear(
            num_layers * hidden_size * self.multiplier,
            latent_size,
        )
        self.hidden2logv = nn.Linear(
            num_layers * hidden_size * self.multiplier,
            latent_size,
        )

        self.latent2hidden = nn.Linear(
            latent_size,
            num_layers * hidden_size,
        )

        self.decoder = nn.GRU(
            embedding_size + latent_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )

        self.reconstruction_loss = nn.CrossEntropyLoss(
            ignore_index=pad_index, reduction="sum"
        )
        self.KL_loss = lambda mean, logv: -0.5 * torch.sum(
            1 + logv - mean.pow(2) - logv.exp()
        )
        self.KL_weight = 0.0
        self.t = 0
        self.step = 5000
        self.k = 0.0025

        self.softmax = nn.Softmax(dim=-1)

        self.save_hyperparameters()

    def loss(self, x, y, mean, logv):
        x = x.swapaxes(1, 2)
        return self.reconstruction_loss(x, y), self.KL_loss(mean, logv)

    def embed(self, X):
        B = X.shape[0]
        # (B, L, E)
        embedded_input = self.embedding(X)

        # (K, B, H)
        _, hidden = self.encoder(embedded_input)
        # (B, K, H)
        hidden = hidden.swapaxes(0, 1)
        # (B, K*H)
        hidden = hidden.reshape(B, self.num_layers * self.hidden_size * self.multiplier)

        # (B, J)
        mean = self.hidden2mean(hidden)
        # (B, J)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([B, self.latent_size]).to(self.device)
        z = z * std + mean

        return z

    def embed2dist(self, X):
        B = X.shape[0]
        # (B, L, E)
        embedded_input = self.embedding(X)

        # (K, B, H)
        _, hidden = self.encoder(embedded_input)
        # (B, K, H)
        hidden = hidden.swapaxes(0, 1)
        # (B, K*H)
        hidden = hidden.reshape(B, self.num_layers * self.hidden_size * self.multiplier)

        # (B, J)
        mean = self.hidden2mean(hidden)
        # (B, J)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        return mean, std

    # (B, L)
    def forward(self, X):
        B = X.shape[0]
        L = X.shape[1]
        # (B, L, E)
        embedded_input = self.embedding(X)

        # (K, B, H)
        _, hidden = self.encoder(embedded_input)
        # (B, K, H)
        hidden = hidden.swapaxes(0, 1)
        # (B, K*H)
        hidden = hidden.reshape(B, self.num_layers * self.hidden_size * self.multiplier)

        # (B, J)
        mean = self.hidden2mean(hidden)
        # (B, J)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([B, self.latent_size]).to(self.device)
        z = z * std + mean

        # (B, K*H)
        hidden = self.latent2hidden(z)
        # (B, K, H)
        hidden = hidden.reshape(B, self.num_layers, self.hidden_size)
        # (K, B, H)
        hidden = hidden.swapaxes(0, 1).contiguous()

        # append z to inputs at each timestep
        new_z = z.repeat(1, L).reshape(B, L, self.latent_size)
        concat_input = torch.cat((embedded_input, new_z), dim=-1)

        # (B, L, H)
        outputs, _ = self.decoder(concat_input, hidden)

        # (B, L, V)
        logits = F.linear(outputs, self.embedding.weight)

        return logits, mean, logv

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat, mean, logv = self.forward(x)
        loss, kl = self.loss(x_hat, y, mean, logv)

        self.KL_weight = float(1 / (1 + np.exp(-self.k * (self.t - self.step))))
        self.t += 1

        self.log("train/train_loss", self.KL_weight * kl + loss)
        self.log("train/train_reconstruction_loss", loss)
        self.log("train/train_kl_divergence", kl)
        self.log("train/train_kl_weight", self.KL_weight)
        return self.KL_weight * kl + loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat, mean, logv = self.forward(x)
        loss, kl = self.loss(x_hat, y, mean, logv)
        self.log("hp_metric", self.KL_weight * kl + loss)
        self.log("val/val_loss", self.KL_weight * kl + loss)
        self.log("val/val_reconstruction_loss", loss)
        self.log("val/val_kl_divergence", kl)
        self.log("val/val_kl_weight", self.KL_weight)
        return self.KL_weight * kl + loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x_hat, mean, logv = self.forward(x)
        self.log("test/test_loss", self.KL_weight * kl + loss)
        self.log("test/test_reconstruction_loss", loss)
        self.log("test/test_kl_divergence", kl)
        self.log("test/test_kl_weight", self.KL_weight)
        return self.KL_weight * kl + loss

    def inference(
        self,
        sos_idx,
        eos_idx,
        max_len=100,
        mode="greedy",
        temperature=1,
        z=None,
        PK=0.9,
        prompt: list = None,
    ):
        if z is None:
            z = torch.randn([1, self.latent_size]).to(self.device)

        new_z = z.repeat(1, max_len).reshape(1, max_len, self.latent_size)

        # (B, K*H)
        hidden = self.latent2hidden(z)
        # (B, K, H)
        hidden = hidden.reshape(1, self.num_layers, self.hidden_size)
        # (K, B, H)
        hidden = hidden.swapaxes(0, 1).contiguous()

        with torch.no_grad():
            if prompt is None:
                generation = [sos_idx]
            else:
                generation = prompt
            t = 0
            while t < max_len:
                input = torch.LongTensor(generation).to(self.device).unsqueeze(0)

                embedded_input = self.embedding(input)

                # concat
                concat_input = torch.cat((embedded_input, new_z[:, : t + 1, :]), dim=-1)

                # (B, L, H)
                outputs, _ = self.decoder(concat_input, hidden)
                outputs = outputs[:, -1, :]

                # (B, L, V)
                logits = F.linear(outputs, self.embedding.weight)

                tok = self.sample(logits, mode=mode, T=temperature, PK=PK)
                generation.append(tok)

                if tok == eos_idx:
                    break
                t += 1

            return generation

    def sample(self, out, mode="greedy", T=1, PK=0.9):
        if mode == "greedy":
            sample = torch.argmax(out)

        elif mode == "topk":
            values, indexes = torch.topk(out, PK, dim=-1)
            out = out.clone().squeeze(1)
            out[out < values[:, -1]] = -float("Inf")
            probs = self.softmax(out / T).squeeze()
            sample = torch.multinomial(probs, 1)

        elif mode == "topp":
            values, indexes = torch.sort(out / T, descending=True)
            values = self.softmax(values)
            cum_probs = torch.cumsum(values, dim=-1)

            remove = cum_probs > PK
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0

            out = out.clone()
            remove = torch.zeros_like(out, dtype=torch.bool).scatter_(
                dim=-1, index=indexes, src=remove
            )
            out[remove] = -float("Inf")

            probs = self.softmax(out / T).squeeze()
            sample = torch.multinomial(probs, 1)

        return sample.item()

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.layer, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


"""
B = 4
L = 5
E = 6
J = 3
z = torch.randint(low=-5, high=-1, size=(B, J))
inp = torch.randint(low=0, high=10, size=(B, L, E))
print(z.shape)
print(z)
print(inp.shape)
print(inp)
new_z = z.repeat(1, L).reshape(B, L, J)
print(new_z.shape)
print(new_z)
new_in = torch.cat((inp, new_z), dim=-1)
print(new_in.shape)
print(new_in)
"""
