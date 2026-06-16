import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# ── ARCHITECTURE ─────────────────────────────────────────────────────────────────
INPUT_DIM  = 14
HIDDEN_DIM = 64
LATENT_DIM = 8

class SpectralVAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   
            nn.GELU(),                  
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h          = self.encoder(x)
        mu         = self.fc_mu(h)
        logvar     = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar


def elbo_loss(recon, x, mu, logvar, beta=0.5):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ── TRAINING LOOP ────────────────────────────────────────────────────────────────
def train_vae(
    model, train_loader, val_loader,
    mean, std,
    n_epochs=50,
    lr=1e-3,
    device='cpu',
    beta=0.5,
):
    model = model.to(device)
    mean, std = mean.to(device), std.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_recon': [], 'val_kl': []}

    for epoch in range(1, n_epochs + 1):
        # ── TRAIN ──
        model.train()
        train_losses = []
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            x_norm  = (x_batch - mean) / std      # z-score normalize per feature

            optimizer.zero_grad()
            recon, mu, logvar = model(x_norm)
            loss, _, _        = elbo_loss(recon, x_norm, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── VALIDATE ──
        model.eval()
        val_losses, val_recons, val_kls = [], [], []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                x_norm  = (x_batch - mean) / std
                recon, mu, logvar      = model(x_norm)
                loss, recon_l, kl_l    = elbo_loss(recon, x_norm, mu, logvar, beta)
                val_losses.append(loss.item())
                val_recons.append(recon_l.item())
                val_kls.append(kl_l.item())

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        v_recon = np.mean(val_recons)
        v_kl    = np.mean(val_kls)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_recon'].append(v_recon)
        history['val_kl'].append(v_kl)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train: {t_loss:.5f} | "
                  f"Val: {v_loss:.5f} (recon={v_recon:.5f}, kl={v_kl:.5f}) | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    return history


# ── ANOMALY SCORING ──────────────────────────────────────────────────────────────

def compute_anomaly_scores(model, loader, mean, std, device='cpu'):
    model.eval()
    mean, std = mean.to(device), std.to(device)

    all_scores  = []
    all_labels  = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            x_norm  = (x_batch - mean) / std
            recon, mu, _ = model(x_norm)

            # Per-pixel MSE across all 14 features
            scores = F.mse_loss(recon, x_norm, reduction='none').mean(dim=1)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(y_batch.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)