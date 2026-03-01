from torch import nn
import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np

class Encoder(nn.Module):
    """
    4-layer CNN encoder
    """

    def __init__(self, input_shape, latent_dim):
        """
        """

        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(self.latent_dim)

        return

    def forward(self, x):
      """
      """

      out = self.convs(x)
      out = self.flatten(out)
      out = self.fc(out)

      return out
    
class VariationalEncoder(Encoder):
    """
    """

    def __init__(self, input_shape, latent_dim):
        """
        """

        super().__init__(input_shape, latent_dim)
        self.fc = nn.LazyLinear(self.latent_dim * 2)

        return

    def forward(self, x):
        """
        """

        out = self.convs(x)
        out = self.flatten(out)
        out = self.fc(out)
        mu, log_var = torch.chunk(out, 2, dim=1)

        return mu, log_var
    
class Decoder(nn.Module):
    """
    """

    def __init__(self, latent_dim, output_shape, base_size=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.base_size = base_size
        self.relu = nn.ReLU()
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.LazyLinear(self.base_size * self.base_size * 256)

        return

    def forward(self, z):
        """
        """

        out = self.fc(z)
        out = self.relu(out)
        out = out.reshape(z.size(0), 256, self.base_size, self.base_size) # Reshape linear output
        out = self.deconvs(out)
        out = self.sigmoid(out)

        return out
    
class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss use MSE
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, model):
        """
        """

        out = model(x, return_params=False)
        loss = F.mse_loss(out, x)

        return loss
    
class VAELoss(nn.Module):
    """
    Reconstruction loss + KL Divergence
    """

    def __init__(self):
        super().__init__()
        return

    def forward(self, x, model, beta=1.0):
        """
        """

        out, (mu, log_var) = model(x, return_params=True)
        rec_loss = F.mse_loss(out, x, reduction="none")
        rec_loss = rec_loss.view(x.size(0), -1).sum(dim=1).mean()
        kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=1)
        kl_loss = kl_per_sample.mean()
        loss = rec_loss + beta * kl_loss

        return loss

class AutoEncoderModel(nn.Module):
    """
    """

    def __init__(self, variational, latent_size, input_shape=(3, 32, 32), base_size=4):
        """
        """

        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        self.variational = variational
        if variational:
            self.encoder = VariationalEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape, base_size)

    def forward(self, x, return_params=False):
        """
        x: (B, C, H, W)
        """

        if self.variational:
            mu, log_var = self.encoder(x)
            std = torch.exp(0.5 * log_var) # (B, L)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = self.encoder(x)
            mu, log_var = None, None
        out = self.decoder(z)

        #
        if return_params:
            return out, (mu, log_var)
        else:
            return out

class BetaScheduler(nn.Module):
    """
    """

    def __init__(self, max_epochs, min_beta=0.01, max_beta=1.0, warmup_fraction=0.2):
        """
        """

        super().__init__()
        self.max_epochs = max_epochs
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.warmup_fraction = warmup_fraction

        return
    
    def forward(self, i_epoch):
        """
        """

        n_epochs = self.warmup_fraction * self.max_epochs
        t = min((i_epoch + 1) / n_epochs, 1.0)
        beta = self.min_beta + t * (self.max_beta - self.min_beta)

        return beta
        
def demo_with_digits(max_iters=200, lr=0.0003, batch_size=32, figsize=(8.5, 2.8)):
    """
    Train a VAE to generate new handwritten digits
    """

    #
    digits = load_digits()
    X = torch.from_numpy(digits.data).to(dtype=torch.float32).reshape(
        -1, 8, 8
    ).unsqueeze(1).repeat(1, 3, 1, 1)
    X = (X / 16.0) # Rescale 0 to 1
    y = torch.from_numpy(digits.target).to(dtype=torch.long)
    ds = TensorDataset(X, y)
    dataloader = DataLoader(ds, batch_size=batch_size)

    #
    vae = AutoEncoderModel(
        variational=True,
        latent_size=16,
        input_shape=(3, 8, 8),
        base_size=1
    )
    loss_fn = VAELoss()
    optimizer = optim.AdamW(vae.parameters(), lr=lr)
    scheduler = BetaScheduler(max_iters, min_beta=0.01, max_beta=1.0, warmup_fraction=0.3)

    #
    vae.train()
    for i_epoch in range(max_iters):
        beta = scheduler(i_epoch)
        batch_loss = 0.0
        for X_b, _ in dataloader:
            loss = loss_fn(X_b, vae, beta=beta)
            batch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_loss /= len(dataloader)
        print(f"Epoch {i_epoch + 1} out of {max_iters}: beta = {beta:.3f}, loss = {batch_loss:.4f}")

    #
    N = X.size()[0]
    indices = np.random.choice(np.arange(0, N), 10, replace=False)
    X_true = X[indices, ...]
    X_pred = vae(X_true).permute(0, 2, 3, 1).detach().cpu().numpy()
    X_true = X_true.permute(0, 2, 3, 1).detach().cpu().numpy()

    #
    z = torch.randn(10, vae.latent_size, device="cuda" if torch.cuda.is_available() else "cpu")
    X_gen = vae.decoder(z)
    X_gen = X_gen.permute(0, 2, 3, 1).detach().cpu()
    X_gen = X_gen.clamp(0, 1).numpy()

    #
    fig, axs = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True)
    for i in range(10):
        axs[0, i].imshow(X_true[i])
        axs[1, i].imshow(X_pred[i])
        axs[2, i].imshow(X_gen[i])

    #
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)

    #
    axs[0, 0].set_ylabel(r"$X_{True}$")
    axs[1, 0].set_ylabel(r"$X_{Recon.}$")
    axs[2, 0].set_ylabel(r"$X_{Gen.}$")

    #
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()

    return fig, axs