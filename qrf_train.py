import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

from qrf_model import (
    RFConfig, QuantumVelocityNet, SequentialQuantumVelocityNet,
    prepare_mnist, images_to_prob, prob_to_image,
    sample_noise_prob, random_hermitian_batch, haar_noise,
    ot_pair, prob_to_sphere, sphere_to_prob, slerp, jacobi_weight,
    load_digit_datasets,
)


def compute_ink_stats(
    cfg: RFConfig, dataset: torch.utils.data.Dataset,
) -> torch.Tensor:
    dt = cfg.torch_dtype
    sums = torch.zeros(cfg.num_classes, dtype=dt)
    sq_sums = torch.zeros(cfg.num_classes, dtype=dt)
    counts = torch.zeros(cfg.num_classes, dtype=dt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    for imgs, labels in loader:
        imgs_small = prepare_mnist(imgs, cfg.img_size).to(dt)
        _, ink = images_to_prob(imgs_small, cfg)
        for c in range(cfg.num_classes):
            mask = labels == c
            if mask.any():
                sums[c] += ink[mask].sum()
                sq_sums[c] += (ink[mask] ** 2).sum()
                counts[c] += mask.sum()
    means = sums / counts.clamp(min=1)
    stds = torch.sqrt(sq_sums / counts.clamp(min=1) - means ** 2).clamp(min=1e-4)
    return torch.stack([means, stds], dim=1)



def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


INIT_SCALE_SKIP_SUBSTRINGS = ("lcu_eta",)


def _should_skip_init_scale(param_name: str) -> bool:
    return any(s in param_name for s in INIT_SCALE_SKIP_SUBSTRINGS)


def apply_init_scale(model, scale: float):
    if scale == 1.0:
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if _should_skip_init_scale(n):
                continue
            p.data *= scale


@torch.no_grad()
def calibrate_init_scale(model, cfg, dataset, device, scales=None):
    dt = cfg.torch_dtype
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(12345)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    imgs, labels = next(iter(loader))
    imgs_s = prepare_mnist(imgs, cfg.img_size).to(dtype=dt, device=device)
    labels = labels.to(device)
    x, _ = images_to_prob(imgs_s, cfg)
    B = x.shape[0]
    eps = cfg.eps

    orig_state = {n: p.data.clone() for n, p in model.named_parameters()}
    model.eval()

    if cfg.prediction_mode == "tangent":
        s1 = prob_to_sphere(x, eps)
        x0 = sample_noise_prob(B, cfg, device, dt)
        s0 = prob_to_sphere(x0, eps)
        t_half = 0.5 * torch.ones(B, dtype=dt, device=device)
        s_t = slerp(s0, s1, t_half, eps)
        x_t = sphere_to_prob(s_t, eps)

        true_delta = s1 - s_t
        true_delta = true_delta - (true_delta * s_t).sum(dim=1, keepdim=True) * s_t
        true_dir = true_delta / true_delta.norm(dim=1, keepdim=True).clamp(min=eps)
        dot_target = (s_t * s1).sum(dim=1).clamp(-1 + eps, 1 - eps)
        theta = torch.acos(dot_target)

        if scales is None:
            scales = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

        best_scale = 0.05
        best_loss = float('inf')

        for scale in scales:
            for n, p in model.named_parameters():
                if _should_skip_init_scale(n):
                    p.data.copy_(orig_state[n])
                else:
                    p.data.copy_(orig_state[n] * scale)

            out = model(x_t, t_half, labels)
            s_out = prob_to_sphere(out, eps)
            v = s_out - s_t
            v = v - (v * s_t).sum(dim=1, keepdim=True) * s_t
            v_norm = v.norm(dim=1, keepdim=True).clamp(min=eps)
            v_dir = v / v_norm
            s1_pred = torch.cos(v_norm) * s_t + torch.sin(v_norm) * v_dir
            loss = (s1_pred - s1).pow(2).sum(dim=1).mean().item()

            if loss < best_loss:
                best_loss = loss
                best_scale = scale

        for n, p in model.named_parameters():
            p.data.copy_(orig_state[n])
        torch.random.set_rng_state(rng_state)
        print(f"  init_scale={best_scale}  (tangent loss={best_loss:.4f})")
        return best_scale

    elif cfg.prediction_mode == "fisher_flow":
        s1 = prob_to_sphere(x, eps)
        x0 = sample_noise_prob(B, cfg, device, dt)
        s0 = prob_to_sphere(x0, eps)
        t_half = 0.5 * torch.ones(B, dtype=dt, device=device)
        s_t = slerp(s0, s1, t_half, eps)
        x_t = sphere_to_prob(s_t, eps)

        dot_1 = (s_t * s1).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
        theta_1 = torch.acos(dot_1)
        sin_1 = torch.sin(theta_1).clamp(min=eps)
        log_1 = theta_1 * (s1 - dot_1 * s_t) / sin_1
        v_target = log_1 / 0.5

        if scales is None:
            scales = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

        best_scale = 0.05
        best_loss = float('inf')

        for scale in scales:
            for n, p in model.named_parameters():
                if _should_skip_init_scale(n):
                    p.data.copy_(orig_state[n])
                else:
                    p.data.copy_(orig_state[n] * scale)

            out = model(x_t, t_half, labels)
            s_out = prob_to_sphere(out, eps)
            dot_out = (s_t * s_out).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
            theta_out = torch.acos(dot_out)
            sin_out = torch.sin(theta_out).clamp(min=eps)
            v_pred = theta_out * (s_out - dot_out * s_t) / sin_out
            loss = (v_pred - v_target).pow(2).sum(dim=1).mean().item()

            if loss < best_loss:
                best_loss = loss
                best_scale = scale

        for n, p in model.named_parameters():
            p.data.copy_(orig_state[n])
        torch.random.set_rng_state(rng_state)
        print(f"  init_scale={best_scale}  (fisher_flow loss={best_loss:.4f})")
        return best_scale

    else:
        target_p = x.clamp(min=1e-12)
        target_p = target_p / target_p.sum(dim=1, keepdim=True)
        target_purity = (target_p ** 2).sum(dim=1).mean().item()
        target_participation = 1.0 / target_purity

        t_half = 0.5 * torch.ones(B, dtype=dt, device=device)

        if scales is None:
            scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]

        best_scale = 1.0
        best_gap = float('inf')

        for scale in scales:
            for n, p in model.named_parameters():
                if _should_skip_init_scale(n):
                    p.data.copy_(orig_state[n])
                else:
                    p.data.copy_(orig_state[n] * scale)

            out = model(x, t_half, labels)
            p_out = out.clamp(min=1e-12)
            p_out = p_out / p_out.sum(dim=1, keepdim=True)
            purity = (p_out ** 2).sum(dim=1).mean().item()
            participation = 1.0 / purity
            gap = abs(participation - target_participation)

            if gap < best_gap:
                best_gap = gap
                best_scale = scale

        for n, p in model.named_parameters():
            p.data.copy_(orig_state[n])
        torch.random.set_rng_state(rng_state)
        print(f"  init_scale={best_scale}  (target participation={target_participation:.1f})")
        return best_scale


def train_mnist_flow(
    cfg: RFConfig, seed: int = 0, device: Optional[torch.device] = None,
) -> Tuple[QuantumVelocityNet, RFConfig, Dict[str, List[float]]]:
    dt = cfg.torch_dtype
    torch.set_default_dtype(dt)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    cfg = cfg.build()

    ds_train, ds_val = load_digit_datasets(cfg.data_source, target_size=cfg.img_size)
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=2 if device.type == "cuda" else 0, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
        num_workers=2 if device.type == "cuda" else 0, pin_memory=device.type == "cuda")

    if cfg.n_stages > 1:
        model = SequentialQuantumVelocityNet(cfg, n_stages=cfg.n_stages).to(device)
    else:
        model = QuantumVelocityNet(cfg).to(device)

    init_scale = calibrate_init_scale(model, cfg, ds_train, device)
    apply_init_scale(model, init_scale)

    if cfg.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adagrad":
        opt = optim.Adagrad(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "riemannian_adam":
        from riemannian_adam import RiemannianAdam
        skip = {"marginal_masks", "hg_perm", "hg_inv_perm"}
        circular_ids = set()
        for name, p in model.named_parameters():
            if any(s in name for s in skip):
                continue
            circular_ids.add(id(p))
        opt = RiemannianAdam(model.parameters(), lr=cfg.lr,
                             betas=(0.9, 0.999), eps=cfg.adam_eps,
                             weight_decay=cfg.weight_decay,
                             circular_param_names=circular_ids)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.lr,
                         betas=(0.9, 0.999), eps=cfg.adam_eps, weight_decay=cfg.weight_decay)

    batches_per_epoch = len(train_loader)
    warmup_steps = cfg.warmup_epochs * batches_per_epoch
    total_steps = cfg.epochs * batches_per_epoch

    if cfg.schedule == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: cosine_warmup_lr(step, warmup_steps, total_steps))
    elif cfg.schedule == "cyclic":
        cycle_steps = cfg.cycle_epochs * batches_per_epoch
        min_frac = 0.1
        def cyclic_lr(step):
            pos = step % cycle_steps
            half = cycle_steps / 2
            if pos < half:
                frac = pos / half
            else:
                frac = 1.0 - (pos - half) / half
            return min_frac + (1.0 - min_frac) * frac
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=cyclic_lr)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)))

    ink_stats = compute_ink_stats(cfg, ds_train)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_qubits={cfg.n_qubits}  dim={cfg.dim}  n_heads={cfg.n_heads}  "
          f"swin_blocks={cfg.n_swin_blocks}  local_layers={cfg.n_local_layers}  "
          f"n_stages={cfg.n_stages} "
          f"total_params={n_params}  device={device}  dtype={cfg.dtype}")
    if cfg.focal_gamma > 0 or cfg.marginal_weight > 0:
        print(f"  focal_gamma={cfg.focal_gamma}  marginal_weight={cfg.marginal_weight}")

    uniform_val = 1.0 / cfg.dim

    nq = cfg.n_qubits
    marginal_masks = torch.zeros(nq, cfg.dim, dtype=dt, device=device)
    for q in range(nq):
        for i in range(cfg.dim):
            if (i >> q) & 1:
                marginal_masks[q, i] = 1.0

    stats: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    def batch_loss(imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        imgs_small = prepare_mnist(imgs, cfg.img_size).to(dtype=dt, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        B = imgs_small.shape[0]

        x1, _ = images_to_prob(imgs_small, cfg)
        s1 = prob_to_sphere(x1, cfg.eps)

        if cfg.noise_type == "haar":
            t = torch.rand(B, dtype=dt, device=device)
            amp1 = torch.sqrt(x1.clamp(min=cfg.eps))
            amp1 = amp1 / (amp1.norm(dim=1, keepdim=True) + cfg.eps)
            H = random_hermitian_batch(B, cfg.dim, dtype=dt, device=device,
                                       scale=cfg.haar_scale)
            x_t = haar_noise(amp1, t, H)
            x_t = x_t / (x_t.sum(dim=1, keepdim=True) + cfg.eps)
        else:
            x0 = sample_noise_prob(B, cfg, device, dtype=dt)
            s0 = prob_to_sphere(x0, cfg.eps)
            if cfg.use_ot_pairing:
                perm = ot_pair(s0, s1)
                x0 = x0[perm]
                s0 = s0[perm]
            s_raw = torch.randn((B,), dtype=dt, device=device)
            t = torch.sigmoid(s_raw)
            if cfg.flow_space == "sphere":
                s_t = slerp(s0, s1, t, cfg.eps)
                x_t = sphere_to_prob(s_t, cfg.eps)
            else:
                x_t = (1.0 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1

        x1_hat = model(x_t, t, labels).to(dt)

        s_out = prob_to_sphere(x1_hat, cfg.eps)
        s_t = prob_to_sphere(x_t, cfg.eps)

        if cfg.prediction_mode == "tangent":
            pred_v = s_out - s_t
            pred_v = pred_v - (pred_v * s_t).sum(dim=1, keepdim=True) * s_t
            pred_norm = pred_v.norm(dim=1, keepdim=True).clamp(min=cfg.eps)
            pred_dir = pred_v / pred_norm

            dot = (s_t * s1).sum(dim=1, keepdim=True).clamp(-1 + cfg.eps, 1 - cfg.eps)
            theta = torch.acos(dot)

            s1_pred = torch.cos(pred_norm) * s_t + torch.sin(pred_norm) * pred_dir
            dot_pred = (s1_pred * s1).sum(dim=1).clamp(-1 + cfg.eps, 1 - cfg.eps)
            loss = torch.acos(dot_pred).pow(2).mean()

            if cfg.diversity_weight > 0.0:
                B_ = s1_pred.shape[0]
                same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(loss.dtype)
                same_class = same_class - torch.eye(B_, dtype=loss.dtype, device=loss.device)
                gram = s1_pred @ s1_pred.T
                denom = same_class.sum().clamp(min=1.0)
                div_loss = (gram * same_class).sum() / denom
                loss = loss + cfg.diversity_weight * div_loss
        elif cfg.prediction_mode == "fisher_flow":
            dot_out = (s_t * s_out).sum(dim=1, keepdim=True).clamp(-1 + cfg.eps, 1 - cfg.eps)
            theta_out = torch.acos(dot_out)
            sin_out = torch.sin(theta_out).clamp(min=cfg.eps)
            v_pred = theta_out * (s_out - dot_out * s_t) / sin_out

            dot_1 = (s_t * s1).sum(dim=1, keepdim=True).clamp(-1 + cfg.eps, 1 - cfg.eps)
            theta_1 = torch.acos(dot_1)
            sin_1 = torch.sin(theta_1).clamp(min=cfg.eps)
            log_1 = theta_1 * (s1 - dot_1 * s_t) / sin_1
            t_safe = t.unsqueeze(1).clamp(max=0.95)
            v_target = log_1 / (1.0 - t_safe)

            loss = (v_pred - v_target).pow(2).sum(dim=1).mean()

            if cfg.diversity_weight > 0.0:
                step = (1.0 - t_safe) * theta_out
                v_dir = v_pred / (v_pred.norm(dim=1, keepdim=True).clamp(min=cfg.eps))
                s1_pred = torch.cos(step) * s_t + torch.sin(step) * v_dir
                B_ = s1_pred.shape[0]
                same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(loss.dtype)
                same_class = same_class - torch.eye(B_, dtype=loss.dtype, device=loss.device)
                gram = s1_pred @ s1_pred.T
                denom = same_class.sum().clamp(min=1.0)
                div_loss = (gram * same_class).sum() / denom
                loss = loss + cfg.diversity_weight * div_loss
        elif cfg.loss_type == "hellinger":
            s_hat = s_out
            s_target = s1

            if cfg.focal_gamma > 0:
                per_pixel_sq_err = (s_hat - s_target) ** 2
                s_centroid = 1.0 / math.sqrt(cfg.dim)
                deviation = (s_target - s_centroid).abs()
                weight = deviation ** cfg.focal_gamma
                weight = weight / (weight.mean(dim=1, keepdim=True) + cfg.eps)
                loss = (weight * per_pixel_sq_err).mean()
            else:
                loss = F.mse_loss(s_hat, s_target, reduction='mean')

            if cfg.use_jacobi and cfg.flow_space == "sphere" and cfg.noise_type != "haar":
                jw = jacobi_weight(t, s0, s1, cfg.eps)
                s_err = (s_hat - s_target) ** 2
                loss = (jw.unsqueeze(1) * s_err).mean()
                if cfg.focal_gamma > 0:
                    s_centroid = 1.0 / math.sqrt(cfg.dim)
                    deviation = (s_target - s_centroid).abs()
                    fw = deviation ** cfg.focal_gamma
                    fw = fw / (fw.mean(dim=1, keepdim=True) + cfg.eps)
                    loss = (jw.unsqueeze(1) * fw * s_err).mean()

            if cfg.marginal_weight > 0:
                target_marginals = x1 @ marginal_masks.T
                pred_marginals = x1_hat @ marginal_masks.T
                marg_loss = F.mse_loss(pred_marginals, target_marginals, reduction='mean')
                loss = loss + cfg.marginal_weight * marg_loss
        elif cfg.loss_type == "kl":
            loss = (x1 * (torch.log(x1 + cfg.eps)
                          - torch.log(x1_hat + cfg.eps))).sum(dim=1).mean()
        elif cfg.loss_type == "cross_entropy":
            loss = -(x1 * torch.log(x1_hat + cfg.eps)).sum(dim=1).mean()
        else:
            loss = F.mse_loss(x1_hat, x1, reduction='mean')

        return loss

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total, n = 0.0, 0
        for imgs, labels in val_loader:
            loss = batch_loss(imgs, labels)
            total += float(loss.cpu()) * imgs.shape[0]
            n += imgs.shape[0]
        return total / max(1, n)

    best_val_loss = float('inf')
    best_state = None

    _first_batch = True
    for epoch in range(cfg.epochs):
        model.train()
        total_loss, nb = 0.0, 0
        for imgs, labels in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(imgs, labels)
            if _first_batch and cfg.compile_sim:
                import time as _time
                print("Compiling backward graph (one-time cost)...")
                _bwd_t0 = _time.perf_counter()
            loss.backward()
            if _first_batch and cfg.compile_sim:
                print(f"  Backward compile complete — {_time.perf_counter()-_bwd_t0:.1f}s")
                _first_batch = False
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            scheduler.step()

            total_loss += float(loss.detach().cpu())
            nb += 1
            if nb % 50 == 0:
                print(f"  epoch={epoch} batch={nb} loss={loss.detach().cpu():.6f}")
        train_loss = total_loss / max(1, nb)
        val_loss = validate()
        current_lr = opt.param_groups[0]['lr']
        stats["train_loss"].append(train_loss)
        stats["val_loss"].append(val_loss)
        stats["lr"].append(current_lr)
        print(f"epoch={epoch}  train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}  lr={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ** new best val_loss={val_loss:.6f} (epoch {epoch})")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.6f})")

    return model, cfg, stats, ink_stats


def _make_time_schedule(N: int, schedule: str, device, dtype) -> torch.Tensor:
    if schedule == "loglinear":
        one_minus_t = torch.logspace(
            0, -math.log10(N), N + 1, device=device, dtype=dtype)
        return 1.0 - one_minus_t
    elif schedule == "cosine":
        i = torch.linspace(0, 1, N + 1, device=device, dtype=dtype)
        return 0.5 * (1.0 - torch.cos(math.pi * i))
    else:
        return torch.linspace(0, 1.0, N + 1, device=device, dtype=dtype)


@torch.no_grad()
def generate_samples(
    model: QuantumVelocityNet, cfg: RFConfig,
    num_samples: int = 32, labels: Optional[List[int]] = None,
    n_steps: Optional[int] = None,
    ink: Optional[torch.Tensor] = None,
    ink_stats: Optional[torch.Tensor] = None,
    integrator: str = "euler",
    time_schedule: str = "loglinear",
) -> torch.Tensor:
    if hasattr(model, 'stages'):
        for stage in model.stages:
            stage.compile_sim = False
    elif hasattr(model, 'compile_sim'):
        model.compile_sim = False

    model.eval()
    device = next(model.parameters()).device
    dt = cfg.torch_dtype
    N = n_steps or cfg.n_flow_steps
    B = num_samples
    eps = cfg.eps

    if labels is None:
        y = torch.randint(0, cfg.num_classes, (B,),
                          device=device, dtype=torch.long)
    else:
        y = torch.tensor(labels[:B], device=device, dtype=torch.long)
        if len(y) < B:
            y = y.repeat(math.ceil(B / len(y)))[:B]

    if cfg.noise_type == "haar":
        gen_alpha = torch.full((cfg.dim,), 1.0, dtype=dt, device=device)
        x0 = torch.distributions.Dirichlet(gen_alpha).sample((B,))
    else:
        x0 = sample_noise_prob(B, cfg, device, dtype=dt)
    x = x0.clone()

    if ink is not None:
        if ink.ndim == 0:
            ink = ink.expand(B).to(device=device, dtype=dt)
        else:
            ink = ink[:B].to(device=device, dtype=dt)
    elif ink_stats is not None:
        ink_stats = ink_stats.to(device=device, dtype=dt)
        means = ink_stats[y, 0]
        stds = ink_stats[y, 1]
        ink = (means + stds * torch.randn(B, device=device,
                                          dtype=dt)).clamp(min=0.0)
    else:
        default_ink = 0.1307 * min(cfg.img_size, 28) ** 2
        ink = torch.full((B,), default_ink,
                         device=device, dtype=dt)

    if cfg.prediction_mode == "tangent":
        s = prob_to_sphere(x, eps)
        max_step = math.pi / (2 * N)
        converge_threshold = max_step * 0.05
        converge_count = torch.zeros(B, dtype=torch.long, device=device)
        converge_patience = 10
        frozen = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(N):
            if frozen.all():
                break
            t_tensor = torch.full((B,), i / N, dtype=dt, device=device)
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
            active = (~frozen).unsqueeze(1).to(dt)
            s_new = torch.cos(step_angle) * s + torch.sin(step_angle) * v_dir
            s = active * s_new + (1 - active) * s
            x = sphere_to_prob(s, eps)
        return prob_to_image(x, ink, cfg)

    if cfg.prediction_mode == "fisher_flow":
        s = prob_to_sphere(x, eps)
        dt_step = 1.0 / N
        for i in range(N):
            t_curr = i / N
            t_tensor = torch.full((B,), t_curr, dtype=dt, device=device)
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
        return prob_to_image(x, ink, cfg)

    if integrator == "geodesic" and cfg.flow_space == "sphere":
        s0 = prob_to_sphere(x, eps)

        t0 = torch.zeros(B, device=device, dtype=dt)
        x1_hat_init = model(x, t0, y)
        s1_init = prob_to_sphere(x1_hat_init, eps)
        dot0 = (s0 * s1_init).sum(dim=1).clamp(-1 + eps, 1 - eps)
        omega_total = torch.acos(dot0)

        for i in range(N):
            s = prob_to_sphere(x, eps)
            dot_to_init = (s * s1_init).sum(dim=1).clamp(-1 + eps, 1 - eps)
            omega_remain_init = torch.acos(dot_to_init)
            t_geo = (1.0 - omega_remain_init / (omega_total + eps)).clamp(0.0, 1.0)

            x1_hat = model(x, t_geo, y)
            s1_hat = prob_to_sphere(x1_hat, eps)

            steps_left = N - i
            frac = (1.0 / steps_left * torch.ones_like(omega_total)).clamp(0.0, 1.0)
            s = slerp(s, s1_hat, frac, eps)
            x = sphere_to_prob(s, eps)

        t_final = torch.ones(B, device=device, dtype=dt)
        x1_hat = model(x, t_final, y)
        return prob_to_image(x1_hat, ink, cfg)

    times = _make_time_schedule(N, time_schedule, device, dt)

    for i in range(N):
        t_curr = float(times[i])
        t_next = float(times[i + 1])
        dt_i = t_next - t_curr
        t_tensor = torch.full((B,), t_curr, device=device, dtype=dt)

        x1_hat = model(x, t_tensor, y)

        if cfg.flow_space == "sphere":
            s = prob_to_sphere(x, eps)
            frac = min(dt_i / (1.0 - t_curr + eps), 1.0)
            frac_t = torch.full((B,), frac, device=device, dtype=dt)

            if integrator == "heun" and frac < 1.0:
                s1_hat = prob_to_sphere(x1_hat, eps)
                s_trial = slerp(s, s1_hat, frac_t, eps)
                x_trial = sphere_to_prob(s_trial, eps)

                t_next_tensor = torch.full(
                    (B,), t_next, device=device, dtype=dt)
                x1_hat2 = model(x_trial, t_next_tensor, y)

                x1_avg = 0.5 * (x1_hat + x1_hat2)
                x1_avg = x1_avg / (x1_avg.sum(dim=1, keepdim=True) + eps)
                s1_avg = prob_to_sphere(x1_avg, eps)
                s = slerp(s, s1_avg, frac_t, eps)
            else:
                s1_hat = prob_to_sphere(x1_hat, eps)
                s = slerp(s, s1_hat, frac_t, eps)

            x = sphere_to_prob(s, eps)
        else:
            v = (x1_hat - x) / (1.0 - t_curr + eps)
            x = x + dt_i * v
            if integrator == "heun" and t_next < 1.0:
                t_next_tensor = torch.full(
                    (B,), t_next, device=device, dtype=dt)
                x1_hat2 = model(x, t_next_tensor, y)
                v2 = (x1_hat2 - x) / (1.0 - t_next + eps)
                x = x - dt_i * v
                x = x + dt_i * 0.5 * (v + v2)
            x = torch.clamp(x, min=0.0)
            x = x / (x.sum(dim=1, keepdim=True) + eps)

    t_final = torch.ones(B, device=device, dtype=dt)
    x1_hat = model(x, t_final, y)
    return prob_to_image(x1_hat, ink, cfg)
