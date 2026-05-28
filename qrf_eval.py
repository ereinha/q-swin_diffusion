import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math

from qrf_model import (
    RFConfig, QuantumVelocityNet, prepare_mnist,
    prob_to_sphere, sphere_to_prob, slerp,
    sample_noise_prob, images_to_prob, prob_to_image,
    load_digit_datasets,
)
from qrf_train import generate_samples


def plot_images(imgs: torch.Tensor, num_rows: int, title: str = ""):
    side = imgs.shape[-1]
    imgs_np = imgs.detach().cpu().reshape(-1, side, side).numpy()
    num_cols = len(imgs_np) // num_rows
    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=(num_cols * 1.5, num_rows * 1.5))
    if num_rows == 1:
        axs = np.atleast_2d(axs)
    if title:
        fig.suptitle(title, fontsize=14)
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(imgs_np):
                axs[i][j].imshow(imgs_np[idx], cmap='gray', vmin=0, vmax=1)
            axs[i][j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_curves(stats: dict):
    has_lr = "lr" in stats and len(stats["lr"]) > 0
    ncols = 2 if has_lr else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].plot(stats["train_loss"], label="Train Loss")
    axes[0].plot(stats["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curves")
    axes[0].legend()

    if has_lr:
        axes[1].plot(stats["lr"], label="Learning Rate", color="tab:orange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def mmd_rbf(X: torch.Tensor, Y: torch.Tensor, sigma: float = 5.0) -> float:
    Y = Y.to(X.dtype)
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2
    denom = 2.0 * sigma * sigma
    return float((torch.exp(-XX / denom).mean()
                  + torch.exp(-YY / denom).mean()
                  - 2.0 * torch.exp(-XY / denom).mean()).item())


@torch.no_grad()
def compute_fid(feats_real: torch.Tensor, feats_gen: torch.Tensor,
                eps: float = 1e-6) -> float:
    mu_r, mu_g = feats_real.double().mean(0), feats_gen.double().mean(0)

    def cov(f):
        f = f.double()
        m = f - f.mean(0, keepdim=True)
        return (m.T @ m) / max(1, f.shape[0] - 1)

    cov_r, cov_g = cov(feats_real), cov(feats_gen)
    diff = mu_r - mu_g

    product = cov_r @ cov_g
    evals, evecs = torch.linalg.eigh(product)
    sqrt_product = (evecs * torch.sqrt(evals.clamp(min=0.0) + eps)) @ evecs.T

    fid = diff.dot(diff) + torch.trace(cov_r + cov_g - 2.0 * sqrt_product)
    return float(fid.clamp(min=0.0).item())


class MNISTClassifier(nn.Module):
    def __init__(self, img_size: int = 8, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        final_size = max(1, img_size // 8)
        self.fc_feat = nn.Linear(128 * final_size * final_size, feat_dim)
        self.fc_out = nn.Linear(feat_dim, 10)

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        feat = self.fc_feat(x)
        logits = self.fc_out(F.relu(feat))
        return (logits, feat) if return_features else logits


_cached_classifier = [None, None, None]


def _get_mnist_classifier(img_size, device, n_epochs=10, data_source="mnist"):
    if (_cached_classifier[0] is not None
            and _cached_classifier[1] == img_size
            and _cached_classifier[2] == data_source):
        return _cached_classifier[0].to(device)

    print(f"  Training classifier for FID (img_size={img_size}, data_source={data_source})...")
    clf = MNISTClassifier(img_size, feat_dim=128).float().to(device)
    ds, _ = load_digit_datasets(data_source, target_size=img_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(clf.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * len(loader))

    clf.train()
    for epoch in range(n_epochs):
        correct, total = 0, 0
        for bx, by in loader:
            bx = bx.float().to(device)
            if bx.ndim == 3:
                bx = bx.unsqueeze(1)
            by = by.to(device)
            opt.zero_grad()
            logits = clf(bx)
            F.cross_entropy(logits, by).backward()
            opt.step()
            sched.step()
            correct += (logits.argmax(1) == by).sum().item()
            total += by.shape[0]
        if epoch == n_epochs - 1:
            print(f"  Classifier accuracy: {correct/total:.4f}")

    clf.eval()
    _cached_classifier[0] = clf
    _cached_classifier[1] = img_size
    _cached_classifier[2] = data_source
    return clf


def evaluate_model(
    model: QuantumVelocityNet, cfg: RFConfig,
    num_gen: int = 1000, num_real: int = 1000,
    ink_stats=None, n_flow_steps: int = None
) -> dict:
    if hasattr(model, 'stages'):
        for stage in model.stages:
            stage.compile_sim = False
    elif hasattr(model, 'compile_sim'):
        model.compile_sim = False

    dt = cfg.torch_dtype
    device = next(model.parameters()).device

    _, ds = load_digit_datasets(cfg.data_source, target_size=cfg.img_size)
    num_real = min(num_real, len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=num_real, shuffle=True)
    real_imgs_full, real_labels = next(iter(loader))
    real_imgs = prepare_mnist(real_imgs_full, cfg.img_size).to(dtype=dt, device="cpu")
    num_real = real_imgs.shape[0]

    gen_labels = (list(range(10)) * (num_gen // 10 + 1))[:num_gen]
    print(f"Generating {num_gen} samples...")
    gen_imgs = generate_samples(model, cfg, num_samples=num_gen, labels=gen_labels,
                                n_steps=n_flow_steps if n_flow_steps is not None else cfg.n_flow_steps,
                                ink_stats=ink_stats,
                                integrator="geodesic")
    gen_imgs = torch.clamp(gen_imgs, 0.0, 1.0).to(dtype=dt, device="cpu")

    print(f"Real Images ({cfg.img_size}x{cfg.img_size}):")
    plot_images(real_imgs[:32], 4, "Real")
    print(f"Generated Images ({cfg.img_size}x{cfg.img_size}):")
    plot_images(gen_imgs[:32], 4, "Generated")

    mmd = mmd_rbf(real_imgs.reshape(num_real, -1),
                  gen_imgs.reshape(num_gen, -1), sigma=3.0)

    mean_l2 = torch.norm(
        (real_imgs[:num_gen].mean(0) - gen_imgs.mean(0)).view(-1), p=2).item()

    clf = _get_mnist_classifier(cfg.img_size, device, data_source=cfg.data_source)

    @torch.no_grad()
    def extract_features(imgs):
        feats = []
        for i in range(0, len(imgs), 256):
            batch = imgs[i:i+256].unsqueeze(1).float().to(device)
            _, f = clf(batch, return_features=True)
            feats.append(f.cpu())
        return torch.cat(feats)

    feats_r = extract_features(real_imgs)
    feats_g = extract_features(gen_imgs)
    fid = compute_fid(feats_r, feats_g)

    logits_g = []
    with torch.no_grad():
        for i in range(0, num_gen, 256):
            batch = gen_imgs[i:i+256].unsqueeze(1).float().to(device)
            logits_g.append(clf(batch).cpu())
    logits_g = torch.cat(logits_g)
    pred = logits_g.argmax(dim=1)
    cond_acc = (pred == torch.tensor(gen_labels[:num_gen])).float().mean().item()

    metrics = {
        "mmd_rbf": mmd,
        "mean_l2": mean_l2,
        "fid": fid,
        "cond_accuracy": cond_acc,
    }
    print(f"\nMetrics: FID={fid:.2f}  cond_acc={cond_acc:.4f}  MMD={mmd:.6f}  L2={mean_l2:.4f}")
    return metrics


@torch.no_grad()
def plot_mnist_flow_trajectory(
    model, cfg: RFConfig,
    digit: int = 3,
    n_steps: int = 60,
    n_snapshots: int = 10,
    n_samples: int = 4,
    ink_stats=None,
    integrator: str = "euler",
):
    if hasattr(model, 'stages'):
        for stage in model.stages:
            stage.compile_sim = False
    elif hasattr(model, 'compile_sim'):
        model.compile_sim = False

    model.eval()
    device = next(model.parameters()).device
    eps = cfg.eps
    B = n_samples

    dt = cfg.torch_dtype
    y = torch.full((B,), digit, device=device, dtype=torch.long)

    if cfg.noise_type == "haar":
        gen_alpha = torch.full((cfg.dim,), 1.0, dtype=dt, device=device)
        x0 = torch.distributions.Dirichlet(gen_alpha).sample((B,))
    else:
        x0 = sample_noise_prob(B, cfg, device, dtype=dt)
    x = x0.clone()

    if ink_stats is not None:
        ink_stats_dev = ink_stats.to(device=device, dtype=cfg.torch_dtype)
        means = ink_stats_dev[y, 0]
        stds = ink_stats_dev[y, 1]
        ink = (means + stds * torch.randn(B, device=device,
                                          dtype=cfg.torch_dtype)).clamp(min=0.0)
    else:
        default_ink = 0.1307 * min(cfg.img_size, 28) ** 2
        ink = torch.full((B,), default_ink, device=device, dtype=cfg.torch_dtype)

    snapshot_indices = np.linspace(0, n_steps - 1, n_snapshots).astype(int)
    snapshots = []
    snapshot_times = []

    dt = 1.0 / n_steps

    if cfg.prediction_mode == "tangent":
        s = prob_to_sphere(x, eps)
        max_step = math.pi / (2 * n_steps)
        converge_threshold = max_step * 0.05
        converge_count = torch.zeros(B, dtype=torch.long, device=device)
        converge_patience = 10
        frozen = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(n_steps):
            if i in snapshot_indices:
                imgs = prob_to_image(x, ink, cfg)
                snapshots.append(imgs.cpu())
                snapshot_times.append(i / n_steps)
            if frozen.all():
                for j in range(i, n_steps):
                    if j in snapshot_indices:
                        imgs = prob_to_image(x, ink, cfg)
                        snapshots.append(imgs.cpu())
                        snapshot_times.append(j / n_steps)
                break
            t_tensor = torch.full((B,), i / n_steps, dtype=cfg.torch_dtype, device=device)
            x1_hat = model(x, t_tensor, y)
            s_out = prob_to_sphere(x1_hat, eps)
            v = s_out - s
            v = v - (v * s).sum(dim=1, keepdim=True) * s
            v_norm = v.norm(dim=1, keepdim=True).clamp(min=eps)
            v_dir = v / v_norm
            step_angle = v_norm.clamp(max=max_step)
            small = (v_norm.squeeze() < converge_threshold)
            converge_count = torch.where(small, converge_count + 1, torch.zeros_like(converge_count))
            frozen = frozen | (converge_count >= converge_patience)
            active = (~frozen).unsqueeze(1).to(cfg.torch_dtype)
            s_new = torch.cos(step_angle) * s + torch.sin(step_angle) * v_dir
            s = active * s_new + (1 - active) * s
            x = sphere_to_prob(s, eps)

    elif cfg.prediction_mode == "fisher_flow":
        s = prob_to_sphere(x, eps)
        dt_step = 1.0 / n_steps
        for i in range(n_steps):
            if i in snapshot_indices:
                imgs = prob_to_image(x, ink, cfg)
                snapshots.append(imgs.cpu())
                snapshot_times.append(i / n_steps)
            t_tensor = torch.full((B,), i / n_steps, dtype=cfg.torch_dtype, device=device)
            x1_hat = model(x, t_tensor, y)
            s_out = prob_to_sphere(x1_hat, eps)

            dot = (s * s_out).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
            theta = torch.acos(dot)
            sin_th = torch.sin(theta).clamp(min=eps)
            v = theta * (s_out - dot * s) / sin_th
            v_norm = v.norm(dim=1, keepdim=True).clamp(min=eps)
            v_dir = v / v_norm

            step_angle = dt_step * v_norm
            s = torch.cos(step_angle) * s + torch.sin(step_angle) * v_dir
            x = sphere_to_prob(s, eps)

    elif integrator == "geodesic" and cfg.flow_space == "sphere":
        s0 = prob_to_sphere(x, eps)
        t0 = torch.zeros(B, device=device, dtype=cfg.torch_dtype)
        x1_hat_init = model(x, t0, y)
        s1_init = prob_to_sphere(x1_hat_init, eps)
        dot0 = (s0 * s1_init).sum(dim=1).clamp(-1 + eps, 1 - eps)
        omega_total = torch.acos(dot0)

        for i in range(n_steps):
            if i in snapshot_indices:
                imgs = prob_to_image(x, ink, cfg)
                snapshots.append(imgs.cpu())
                snapshot_times.append(i / n_steps)

            s = prob_to_sphere(x, eps)
            dot_to_init = (s * s1_init).sum(dim=1).clamp(-1 + eps, 1 - eps)
            omega_remain = torch.acos(dot_to_init)
            t_geo = (1.0 - omega_remain / (omega_total + eps)).clamp(0.0, 1.0)

            x1_hat = model(x, t_geo, y)
            s1_hat = prob_to_sphere(x1_hat, eps)

            steps_left = n_steps - i
            frac = (1.0 / steps_left * torch.ones_like(omega_remain)).clamp(
                0.0, 1.0
            )
            s = slerp(s, s1_hat, frac, eps)
            x = sphere_to_prob(s, eps)

    else:
        for i in range(n_steps):
            if i in snapshot_indices:
                imgs = prob_to_image(x, ink, cfg)
                snapshots.append(imgs.cpu())
                snapshot_times.append(i / n_steps)

            t_val = i * dt
            t = torch.full((B,), t_val, device=device, dtype=cfg.torch_dtype)
            x1_hat = model(x, t, y)

            if cfg.flow_space == "sphere":
                s = prob_to_sphere(x, eps)
                s1_hat = prob_to_sphere(x1_hat, eps)
                frac = min(dt / (1.0 - t_val + eps), 1.0)
                frac_t = torch.full(
                    (B,), frac, device=device, dtype=cfg.torch_dtype
                )
                s = slerp(s, s1_hat, frac_t, eps)
                x = sphere_to_prob(s, eps)
            else:
                v = (x1_hat - x) / (1.0 - t_val + eps)
                x = x + dt * v
                x = torch.clamp(x, min=0.0)
                x = x / (x.sum(dim=1, keepdim=True) + eps)

    t_final = torch.ones(B, device=device, dtype=cfg.torch_dtype)
    x_final = model(x, t_final, y)
    final_imgs = prob_to_image(x_final, ink, cfg)
    snapshots.append(final_imgs.cpu())
    snapshot_times.append(1.0)

    n_snap = len(snapshots)
    fig, axes = plt.subplots(
        B, n_snap, figsize=(1.8 * n_snap, 1.8 * B),
        squeeze=False,
    )

    for row in range(B):
        for col in range(n_snap):
            img = snapshots[col][row].numpy()
            axes[row][col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row][col].axis("off")
            if row == 0:
                if col < n_snap - 1:
                    axes[0][col].set_title(
                        f"t={snapshot_times[col]:.2f}", fontsize=9
                    )
                else:
                    axes[0][col].set_title("Final", fontsize=9)

    fig.suptitle(
        f"Flow Trajectory: Noise -> Digit {digit}  "
        f"({n_steps} steps, {integrator})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()
