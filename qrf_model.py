import copy
from dataclasses import dataclass
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml
except ImportError:
    qml = None

from torch_sim import simulate_qrf_circuit
from torch_sim_real import (
    simulate_qrf_circuit_real, _amp_to_state_real, apply_block_encoding_real,
    apply_block_encoding_complex, hadamard_test_overlap,
)

_compiled_qrf_sim = [None, None]


def _reverse_bits(val, n_bits):
    out = 0
    for i in range(n_bits):
        if val & (1 << i):
            out |= 1 << (n_bits - 1 - i)
    return out


def gray2d_permutation(grid_size, n_bits_per_axis, aligned=False):
    n = grid_size * grid_size
    nb = n_bits_per_axis
    perm = torch.zeros(n, dtype=torch.long)
    for r in range(grid_size):
        gr = r ^ (r >> 1)
        if aligned:
            gr = _reverse_bits(gr, nb)
        for c in range(grid_size):
            gc = c ^ (c >> 1)
            if aligned:
                gc = _reverse_bits(gc, nb)
            flat_idx = r * grid_size + c
            basis = gr | (gc << nb)
            perm[flat_idx] = basis
    assert len(set(perm.tolist())) == n, "2D Gray permutation is not bijective"
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n)
    return perm, inv_perm


def compute_layout(n_bits_per_axis: int) -> dict:
    nq = 2 * n_bits_per_axis
    row_q = list(range(n_bits_per_axis))
    col_q = list(range(n_bits_per_axis, nq))

    def split(macro_row_idx: int, macro_col_idx: int):
        macro = [row_q[macro_row_idx], col_q[macro_col_idx]]
        micro = [q for q in range(nq) if q not in macro]
        return macro, micro

    macro_n, micro_n = split(0, 0)
    macro_s, micro_s = split(1, 1)

    def micro_edges(micro, row_q_set, col_q_set):
        rm = [w for w in micro if w in row_q_set]
        cm = [w for w in micro if w in col_q_set]
        edges = []
        for i in range(len(rm) - 1):
            edges.append((rm[i], rm[i + 1]))
        for i in range(len(cm) - 1):
            edges.append((cm[i], cm[i + 1]))
        for r in rm:
            for c in cm:
                edges.append((r, c))
        return edges

    def cross_edges(macro, micro, row_q_set, col_q_set):
        edges = []
        for m in macro:
            same_axis = row_q_set if m in row_q_set else col_q_set
            for u in micro:
                if u in same_axis:
                    edges.append((m, u))
            cross_axis = col_q_set if m in row_q_set else row_q_set
            for u in micro:
                if u in cross_axis:
                    edges.append((m, u))
        return edges

    row_set, col_set = set(row_q), set(col_q)
    me_n = micro_edges(micro_n, row_set, col_set)
    me_s = micro_edges(micro_s, row_set, col_set)
    ce_n = cross_edges(macro_n, micro_n, row_set, col_set)
    ce_s = cross_edges(macro_s, micro_s, row_set, col_set)

    return dict(
        nq=nq, n_macro=len(macro_n), n_micro=len(micro_n),
        all_wires=list(range(nq)),
        macro_normal=macro_n, micro_normal=micro_n,
        macro_shifted=macro_s, micro_shifted=micro_s,
        micro_edges_normal=me_n, micro_edges_shifted=me_s,
        cross_edges_normal=ce_n, cross_edges_shifted=ce_s,
    )


@dataclass
class RFConfig:
    num_classes: int = 10
    img_size: int = 16
    n_qubits: int = 8
    n_swin_blocks: int = 6
    n_local_layers: int = 3
    n_heads: int = 1
    lr: float = 5e-3
    batch_size: int = 64
    epochs: int = 15
    warmup_epochs: int = 0
    n_flow_steps: int = 50
    noise_alpha: float = 1.0
    device_name: str = "default.qubit"
    diff_method: str = "backprop"
    shots: Optional[int] = None
    eps: float = 1e-8
    flow_space: str = "sphere"
    loss_type: str = "hellinger"
    label_reup_scale: float = 1.0
    intra_reup: bool = True
    optimizer: str = "riemannian_adam"
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    schedule: str = "warmup"
    cycle_epochs: int = 5
    pixel_order: str = "gray2d_aligned"
    focal_gamma: float = 1.0
    marginal_weight: float = 5.0
    grad_clip: float = 5.0
    n_stages: int = 1
    stage_cond: bool = False
    coherent_stages: bool = False
    residual_mixing: bool = False
    use_pennylane: bool = False
    compile_sim: bool = False
    dtype: str = "float64"
    use_jacobi: bool = False
    noise_type: str = "slerp"
    haar_scale: float = 2.0
    use_pairwise_reup: bool = False
    use_rowcol_reup: bool = False
    prediction_mode: str = "x1"
    use_time_input: bool = True
    use_ot_pairing: bool = False
    diversity_weight: float = 0.0
    data_source: str = "mnist"
    lcu_residual: bool = False
    block_encode_proj: bool = False
    block_encode_rank: int = 4
    block_encode_layers: int = 3
    qkv_attention: bool = False
    qkv_layers: int = 3
    soft_lcu: bool = False
    soft_lcu_init: float = 0.5
    dim: int = 0

    @property
    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32

    def build(self) -> "RFConfig":
        self.dim = 2 ** self.n_qubits
        assert self.dim == self.img_size * self.img_size, \
            f"2^n_qubits={self.dim} must equal img_size²={self.img_size**2}"
        return self


def prepare_mnist(imgs: torch.Tensor, target_size: int) -> torch.Tensor:
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(1)
    src_size = imgs.shape[-1]
    if target_size == src_size:
        out = imgs.float()
    elif target_size < src_size:
        out = F.interpolate(imgs.float(), size=(target_size, target_size), mode='area')
    else:
        out = F.interpolate(imgs.float(), size=(target_size, target_size), mode='bilinear',
                            align_corners=False)
    return out.squeeze(1)


class SklearnDigitsDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool = True, val_frac: float = 0.15, seed: int = 42):
        from sklearn.datasets import load_digits
        digits = load_digits()
        imgs = torch.from_numpy(digits.images.astype('float32')) / 16.0
        imgs = imgs.unsqueeze(1)
        labels = torch.from_numpy(digits.target.astype('int64'))
        n = imgs.shape[0]
        n_val = int(round(n * val_frac))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        idx = perm[n_val:] if train else perm[:n_val]
        self.imgs = imgs[idx].contiguous()
        self.labels = labels[idx].contiguous()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]


class _PreResizedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, target_size: int):
        loader = torch.utils.data.DataLoader(source_dataset, batch_size=512, shuffle=False)
        imgs_list, labels_list = [], []
        for imgs, labels in loader:
            imgs_list.append(prepare_mnist(imgs, target_size))
            labels_list.append(labels)
        self.imgs = torch.cat(imgs_list, dim=0).unsqueeze(1).contiguous()
        self.labels = torch.cat(labels_list, dim=0).contiguous()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]


def load_digit_datasets(source: str = "mnist", target_size: Optional[int] = None):
    if source == "mnist":
        import torchvision
        from torchvision import transforms
        ds_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor())
        ds_val = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor())
        if target_size is not None and target_size != 28:
            ds_train = _PreResizedDataset(ds_train, target_size)
            ds_val = _PreResizedDataset(ds_val, target_size)
        return ds_train, ds_val
    if source == "sklearn":
        ds_train = SklearnDigitsDataset(train=True)
        ds_val = SklearnDigitsDataset(train=False)
        if target_size is not None and target_size != 8:
            ds_train = _PreResizedDataset(ds_train, target_size)
            ds_val = _PreResizedDataset(ds_val, target_size)
        return ds_train, ds_val
    raise ValueError(f"unknown data_source={source!r}; expected 'mnist' or 'sklearn'")


def images_to_prob(imgs: torch.Tensor, cfg: RFConfig):
    B = imgs.shape[0]
    flat = imgs.reshape(B, -1).to(imgs.dtype)
    flat = torch.clamp(flat, min=0.0)
    ink = flat.sum(dim=1)
    return flat / (ink.unsqueeze(1) + cfg.eps), ink


def prob_to_image(p: torch.Tensor, ink: torch.Tensor, cfg: RFConfig) -> torch.Tensor:
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if ink.ndim == 0:
        ink = ink.unsqueeze(0)
    p = torch.clamp(p, min=0.0)
    return (p * ink.unsqueeze(1)).clamp(0.0, 1.0).reshape(-1, cfg.img_size, cfg.img_size)


def compute_pairwise_masks(nq, dim, dtype):
    pairs = []
    pair_masks = []
    for q in range(nq):
        for r in range(q + 1, nq):
            mask = torch.zeros(dim, dtype=dtype)
            for i in range(dim):
                bq = (i >> q) & 1
                br = (i >> r) & 1
                mask[i] = 1.0 if (bq == br) else -1.0
            pair_masks.append(mask)
            pairs.append((q, r))
    return pairs, torch.stack(pair_masks)


def compute_row_col_masks(img_size, dim, dtype):
    row_masks = []
    for r in range(img_size):
        mask = torch.zeros(dim, dtype=dtype)
        for c in range(img_size):
            mask[r * img_size + c] = 1.0
        row_masks.append(mask)
    col_masks = []
    for c in range(img_size):
        mask = torch.zeros(dim, dtype=dtype)
        for r in range(img_size):
            mask[r * img_size + c] = 1.0
        col_masks.append(mask)
    return torch.stack(row_masks), torch.stack(col_masks)


@torch.no_grad()
def ot_pair(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    dot = s0 @ s1.T
    cost = torch.acos(dot.clamp(-1 + 1e-6, 1 - 1e-6))
    from scipy.optimize import linear_sum_assignment
    row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())
    return torch.tensor(col_idx, dtype=torch.long, device=s0.device)


def sample_noise_prob(B: int, cfg, device: torch.device,
                      dtype: torch.dtype = torch.float64) -> torch.Tensor:
    alpha = torch.full((cfg.dim,), cfg.noise_alpha, dtype=dtype, device=device)
    return torch.distributions.Dirichlet(alpha).sample((B,))


def random_hermitian_batch(B, D, dtype=torch.float64, device='cpu', scale=2.0):
    re = torch.randn(B, D, D, dtype=dtype, device=device)
    im = torch.randn(B, D, D, dtype=dtype, device=device)
    Z = torch.complex(re, im)
    H = (Z + Z.conj().transpose(-2, -1)) / 2
    norms = torch.linalg.norm(H.reshape(B, -1), dim=1, keepdim=True).unsqueeze(-1)
    return H / norms * math.pi * scale


def haar_noise(amp, t, H_batch):
    B, D = amp.shape
    cdtype = torch.complex128 if amp.dtype == torch.float64 else torch.complex64
    psi = amp.to(cdtype)
    t_expand = t.reshape(B, 1, 1).to(cdtype)
    U = torch.linalg.matrix_exp(1j * t_expand * H_batch)
    psi_out = torch.bmm(U, psi.unsqueeze(-1)).squeeze(-1)
    probs = psi_out.real ** 2 + psi_out.imag ** 2
    return probs.to(amp.dtype)


def prob_to_sphere(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = torch.sqrt(torch.clamp(p, min=eps))
    return s / (s.norm(dim=1, keepdim=True) + eps)


def sphere_to_prob(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = s ** 2
    return p / (p.sum(dim=1, keepdim=True) + eps)


def slerp(s0: torch.Tensor, s1: torch.Tensor, t: torch.Tensor,
          eps: float = 1e-8) -> torch.Tensor:
    dot = (s0 * s1).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    t_2d = t.unsqueeze(1) if t.ndim == 1 else t
    small = (sin_omega.abs() < 1e-6)
    coeff0 = torch.where(small, 1.0 - t_2d,
                         torch.sin((1.0 - t_2d) * omega) / (sin_omega + eps))
    coeff1 = torch.where(small, t_2d,
                         torch.sin(t_2d * omega) / (sin_omega + eps))
    return coeff0 * s0 + coeff1 * s1


def jacobi_weight(t: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    dot = (s0 * s1).sum(dim=1).clamp(-(1.0 - eps), 1.0 - eps)
    Omega = torch.acos(dot)
    arg = t * Omega
    sinc = torch.where(arg.abs() < eps,
                        torch.ones_like(arg),
                        torch.sin(arg) / arg)
    w = sinc ** 2
    return w / w.mean().clamp(min=eps)


def _apply_label_rotations(amp: torch.Tensor, angles: torch.Tensor,
                            n_qubits: int) -> torch.Tensor:
    dim = amp.shape[1]
    state = amp.clone()
    for q in range(n_qubits):
        theta = angles[:, q]
        cos_half = torch.cos(theta / 2).unsqueeze(1)
        sin_half = torch.sin(theta / 2).unsqueeze(1)
        stride = 2 ** q
        block = 2 * stride
        new_state = state.clone()
        for start in range(0, dim, block):
            idx0 = list(range(start, start + stride))
            idx1 = list(range(start + stride, start + block))
            s0 = state[:, idx0]
            s1 = state[:, idx1]
            new_state[:, idx0] = cos_half * s0 - sin_half * s1
            new_state[:, idx1] = sin_half * s0 + cos_half * s1
        state = new_state
    state = state / (state.norm(dim=1, keepdim=True) + 1e-8)
    return state


class QuantumVelocityNet(nn.Module):
    def __init__(self, cfg: RFConfig):
        super().__init__()
        self.cfg = cfg
        nq = cfg.n_qubits
        nb = cfg.n_swin_blocks
        nll = cfg.n_local_layers
        nh = cfg.n_heads
        n_bits = nq // 2

        layout = compute_layout(n_bits)
        self.layout = layout
        n_micro = layout['n_micro']
        n_macro = layout['n_macro']
        n_me = len(layout['micro_edges_normal'])
        n_ce = len(layout['cross_edges_normal'])
        all_wires = layout['all_wires']

        if cfg.pixel_order in ("gray2d", "gray2d_aligned"):
            aligned = (cfg.pixel_order == "gray2d_aligned")
            perm, inv_perm = gray2d_permutation(cfg.img_size, n_bits, aligned=aligned)
            self.register_buffer('hg_perm', perm)
            self.register_buffer('hg_inv_perm', inv_perm)
        else:
            self.register_buffer('hg_perm', None)
            self.register_buffer('hg_inv_perm', None)

        dt = cfg.torch_dtype
        masks = torch.zeros(nq, cfg.dim, dtype=dt)
        for q in range(nq):
            for i in range(cfg.dim):
                if (i >> q) & 1:
                    masks[q, i] = 1.0
        self.register_buffer('marginal_masks', masks)

        self.use_pairwise_reup = cfg.use_pairwise_reup
        self.use_rowcol_reup = cfg.use_rowcol_reup

        if cfg.use_pairwise_reup:
            pairs, pair_masks = compute_pairwise_masks(nq, cfg.dim, dt)
            self.pair_list = pairs
            self.register_buffer('pair_masks', pair_masks)
            n_pairs = len(pairs)
        else:
            self.pair_list = []
            n_pairs = 0

        if cfg.use_rowcol_reup:
            row_masks, col_masks = compute_row_col_masks(cfg.img_size, cfg.dim, dt)
            self.register_buffer('row_masks', row_masks)
            self.register_buffer('col_masks', col_masks)
        n_rc = cfg.img_size

        self._all_wires = all_wires
        self._macro_n = layout['macro_normal']
        self._micro_n = layout['micro_normal']
        self._macro_s = layout['macro_shifted']
        self._micro_s = layout['micro_shifted']
        self._me_n = layout['micro_edges_normal']
        self._me_s = layout['micro_edges_shifted']
        self._ce_n = layout['cross_edges_normal']
        self._ce_s = layout['cross_edges_shifted']

        total_depth = nb * nll
        sq_scale = 0.3 / math.sqrt(total_depth)
        ent_scale = 0.3 / total_depth
        cross_ent_scale = 0.3 / nb
        cross_sq_scale = 0.1 / math.sqrt(nb)
        time_scale = 0.5 / math.sqrt(nb)
        ent_bias = math.pi / 8

        self.local_ry = nn.Parameter(
            sq_scale * torch.randn(nh, nb, nll, n_micro, dtype=dt))
        self.local_rz = nn.Parameter(
            sq_scale * torch.randn(nh, nb, nll, n_micro, dtype=dt))
        self.local_ent = nn.Parameter(
            ent_bias / 2 + ent_scale * torch.randn(nh, nb, nll, n_me, dtype=dt))

        self.cross_zz = nn.Parameter(
            ent_bias + cross_ent_scale * torch.randn(nh, nb, n_ce, dtype=dt))
        self.cross_macro_ry = nn.Parameter(
            cross_sq_scale * torch.randn(nh, nb, n_macro, dtype=dt))

        self.reup_ry = nn.Parameter(
            sq_scale * torch.randn(nh, nb, nq, dtype=dt))
        self.reup_time = nn.Parameter(
            time_scale * torch.randn(nh, nb, nq, dtype=dt))
        self.reup_data = nn.Parameter(
            time_scale * torch.randn(nh, nb, nq, dtype=dt))

        self.intra_reup_ry = nn.Parameter(
            sq_scale * torch.randn(nh, nb, nll, nq, dtype=dt))
        self.intra_reup_time = nn.Parameter(
            time_scale * torch.randn(nh, nb, nll, nq, dtype=dt))
        self.intra_reup_data = nn.Parameter(
            time_scale * torch.randn(nh, nb, nll, nq, dtype=dt))

        if cfg.use_pairwise_reup:
            pair_scale = 0.1 / math.sqrt(total_depth)
            self.pair_reup_rz = nn.Parameter(
                pair_scale * torch.randn(nh, nb, nll, n_pairs, dtype=dt))
            self.pair_reup_data = nn.Parameter(
                pair_scale * torch.randn(nh, nb, nll, n_pairs, dtype=dt))

        if cfg.use_rowcol_reup:
            rc_scale = 0.1 / math.sqrt(total_depth)
            self.rowcol_reup_ry = nn.Parameter(
                rc_scale * torch.randn(nh, nb, nll, nq, dtype=dt))
            self.rowcol_reup_data = nn.Parameter(
                rc_scale * torch.randn(nh, nb, nll, nq, n_rc, dtype=dt))

        post_scale = 0.3 / math.sqrt(nb + 1)
        self.post_ry1 = nn.Parameter(
            post_scale * torch.randn(nh, nq, dtype=dt))
        self.post_rz1 = nn.Parameter(
            post_scale * torch.randn(nh, nq, dtype=dt))
        self.post_ry2 = nn.Parameter(
            post_scale * torch.randn(nh, nq, dtype=dt))
        self.label_embed = nn.Parameter(
            0.3 * torch.randn(cfg.num_classes, nq, dtype=dt))
        self.label_reup = nn.Parameter(
            0.3 * torch.randn(cfg.num_classes, nb, nq, dtype=dt))

        self.residual_mixing = cfg.residual_mixing
        if cfg.residual_mixing:
            self.residual_logits = nn.Parameter(
                torch.zeros(nb, dtype=dt))

        self.block_encode_proj = cfg.block_encode_proj
        if cfg.block_encode_proj:
            L = cfg.block_encode_layers
            be_scale = 0.05
            self.block_ry_layers = nn.Parameter(
                be_scale * torch.randn(nh, L, nq + 1, dtype=dt))
            self.block_cry_anchor = nn.Parameter(
                be_scale * torch.randn(nh, L, nq, dtype=dt))
            self.block_cry_data = nn.Parameter(
                be_scale * torch.randn(nh, L, nq - 1, dtype=dt))
            self.block_ry_final = nn.Parameter(
                be_scale * torch.randn(nh, nq + 1, dtype=dt))

        self.qkv_attention = cfg.qkv_attention
        if cfg.qkv_attention:
            assert not cfg.block_encode_proj, \
                "qkv_attention and block_encode_proj are mutually exclusive"
            assert not cfg.use_pennylane, \
                "qkv_attention requires the torch simulator (use_pennylane=False)"
            assert not cfg.coherent_stages, \
                "qkv_attention requires coherent_stages=False (stage-level residual needs amp_in)"
            Lq = cfg.qkv_layers
            qkv_scale = 0.05
            for prefix in ("v", "q", "k"):
                self.register_parameter(
                    f"qkv_{prefix}_rot_layers",
                    nn.Parameter(qkv_scale * torch.randn(nh, Lq, nq + 1, 3, dtype=dt)))
                self.register_parameter(
                    f"qkv_{prefix}_anchor",
                    nn.Parameter(qkv_scale * torch.randn(nh, Lq, nq, dtype=dt)))
                self.register_parameter(
                    f"qkv_{prefix}_data",
                    nn.Parameter(qkv_scale * torch.randn(nh, Lq, nq - 1, dtype=dt)))
                self.register_parameter(
                    f"qkv_{prefix}_rot_final",
                    nn.Parameter(qkv_scale * torch.randn(nh, nq + 1, 3, dtype=dt)))

        self.soft_lcu = cfg.soft_lcu
        if cfg.soft_lcu:
            assert cfg.lcu_residual, \
                "soft_lcu requires lcu_residual=True"
            self.lcu_eta = nn.Parameter(
                cfg.soft_lcu_init * torch.ones(nh, nb, dtype=dt))

        self.compile_sim = cfg.compile_sim

        self.use_pennylane = cfg.use_pennylane
        self.devs = []
        self.qnodes = []
        if cfg.use_pennylane:
            for _ in range(nh):
                dev = qml.device(cfg.device_name, wires=nq, shots=cfg.shots)
                self.devs.append(dev)

                @qml.qnode(dev, interface="torch", diff_method=cfg.diff_method)
                def circuit(amp, t_val, marginals,
                            local_ry, local_rz, local_ent,
                            cross_zz, cross_macro_ry,
                            reup_ry, reup_time, reup_data,
                            label_angles,
                            intra_reup_ry, intra_reup_time,
                            intra_reup_data,
                            post_ry1, post_rz1, post_ry2,
                            _nb=nb, _nll=nll, _nq=nq,
                            _all=all_wires, _intra=cfg.intra_reup,
                            _macro_n=self._macro_n, _micro_n=self._micro_n,
                            _macro_s=self._macro_s, _micro_s=self._micro_s,
                            _me_n=self._me_n, _me_s=self._me_s,
                            _ce_n=self._ce_n, _ce_s=self._ce_s):

                    qml.AmplitudeEmbedding(amp, wires=_all, normalize=True)

                    for b in range(_nb):
                        if b % 2 == 0:
                            w_micro, micro_edges = _micro_n, _me_n
                            w_macro, cross_edges = _macro_n, _ce_n
                        else:
                            w_micro, micro_edges = _micro_s, _me_s
                            w_macro, cross_edges = _macro_s, _ce_s

                        for ll in range(_nll):
                            for k, w in enumerate(w_micro):
                                qml.RY(local_ry[b, ll, k], wires=w)
                                qml.RZ(local_rz[b, ll, k], wires=w)
                            for k, (wa, wb) in enumerate(micro_edges):
                                qml.CRY(local_ent[b, ll, k], wires=[wa, wb])
                            if _intra and ll < _nll - 1:
                                for q in range(_nq):
                                    qml.RY(intra_reup_ry[b, ll, q]
                                            + intra_reup_time[b, ll, q] * t_val
                                            + intra_reup_data[b, ll, q] * marginals[q],
                                            wires=q)

                        for k, w in enumerate(w_macro):
                            qml.RY(cross_macro_ry[b, k], wires=w)
                        for k, (wa, wb) in enumerate(cross_edges):
                            qml.CRY(cross_zz[b, k], wires=[wa, wb])

                        for q in range(_nq):
                            qml.RY(reup_ry[b, q] + reup_time[b, q] * t_val
                                    + reup_data[b, q] * marginals[q],
                                    wires=q)
                            qml.RX(label_angles[b, q], wires=q)

                    for q in range(_nq):
                        qml.RY(post_ry1[q], wires=q)
                        qml.RZ(post_rz1[q], wires=q)
                    for q in range(_nq):
                        qml.RY(post_ry2[q], wires=q)

                    return qml.probs(wires=_all)

                self.qnodes.append(circuit)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                labels: torch.Tensor,
                label_embed_offset: torch.Tensor = None) -> torch.Tensor:
        cfg = self.cfg
        dt = self.marginal_masks.dtype
        x_t = x_t.to(dt)
        t = t.to(dt)

        if not cfg.use_time_input:
            t = torch.zeros_like(t)

        if self.hg_inv_perm is not None:
            x_t = x_t[:, self.hg_inv_perm]

        amp = torch.sqrt(torch.clamp(x_t, min=cfg.eps))
        amp = amp / (amp.norm(dim=1, keepdim=True) + cfg.eps)

        marginals = ((x_t @ self.marginal_masks.T) - 0.5).T.contiguous()

        pair_marg = (x_t @ self.pair_masks.T).T if self.use_pairwise_reup else None
        row_marg = ((x_t @ self.row_masks.T).T - 1.0 / cfg.img_size) if self.use_rowcol_reup else None

        embed = self.label_embed
        if label_embed_offset is not None:
            embed = embed + label_embed_offset
        label_angles = embed[labels]
        amp = _apply_label_rotations(amp, label_angles, cfg.n_qubits)
        amp = amp.contiguous()

        label_block_angles = (cfg.label_reup_scale * self.label_reup[labels].permute(1, 2, 0)).contiguous()

        head_outputs = []
        for h in range(cfg.n_heads):
            amp_h = amp
            qkv_score = None
            qkv_state_in = None
            if self.block_encode_proj:
                amp_h = apply_block_encoding_real(
                    amp,
                    self.block_ry_layers[h],
                    self.block_cry_anchor[h],
                    self.block_cry_data[h],
                    self.block_ry_final[h],
                    cfg.n_qubits,
                    eps=cfg.eps,
                )
            elif self.qkv_attention:
                re_V, im_V = apply_block_encoding_complex(
                    amp, self.qkv_v_rot_layers[h], self.qkv_v_anchor[h],
                    self.qkv_v_data[h], self.qkv_v_rot_final[h],
                    cfg.n_qubits, eps=cfg.eps)
                re_Q, im_Q = apply_block_encoding_complex(
                    amp, self.qkv_q_rot_layers[h], self.qkv_q_anchor[h],
                    self.qkv_q_data[h], self.qkv_q_rot_final[h],
                    cfg.n_qubits, eps=cfg.eps)
                re_K, im_K = apply_block_encoding_complex(
                    amp, self.qkv_k_rot_layers[h], self.qkv_k_anchor[h],
                    self.qkv_k_data[h], self.qkv_k_rot_final[h],
                    cfg.n_qubits, eps=cfg.eps)
                qkv_state_in = (re_V, im_V)
                z = hadamard_test_overlap(re_Q, im_Q, re_K, im_K, cfg.n_qubits)
                qkv_score = ((1.0 + z) / 2.0).to(dt)
            if self.use_pennylane:
                probs_h = self.qnodes[h](
                    amp_h, t, marginals,
                    self.local_ry[h], self.local_rz[h], self.local_ent[h],
                    self.cross_zz[h], self.cross_macro_ry[h],
                    self.reup_ry[h], self.reup_time[h],
                    self.reup_data[h],
                    label_block_angles,
                    self.intra_reup_ry[h], self.intra_reup_time[h],
                    self.intra_reup_data[h],
                    self.post_ry1[h], self.post_rz1[h],
                    self.post_ry2[h],
                ).to(dt)
            else:
                sim_fn = simulate_qrf_circuit_real
                if self.compile_sim and _compiled_qrf_sim[0] is None:
                    import time as _time
                    torch._dynamo.config.cache_size_limit = 256
                    print("Compiling circuit simulator (one-time cost)...")
                    _t0 = _time.perf_counter()
                    _compiled_qrf_sim[0] = torch.compile(simulate_qrf_circuit_real, fullgraph=False)
                    _compiled_qrf_sim[1] = _t0
                if _compiled_qrf_sim[0] is not None and self.compile_sim:
                    sim_fn = _compiled_qrf_sim[0]
                extra_kwargs = {}
                if self.use_pairwise_reup:
                    extra_kwargs.update(
                        pair_marginals=pair_marg,
                        pair_reup_rz=self.pair_reup_rz[h],
                        pair_reup_data=self.pair_reup_data[h],
                        pair_list=self.pair_list,
                    )
                if self.use_rowcol_reup:
                    extra_kwargs.update(
                        row_marginals=row_marg,
                        rowcol_reup_ry=self.rowcol_reup_ry[h],
                        rowcol_reup_data=self.rowcol_reup_data[h],
                    )
                if qkv_state_in is not None:
                    extra_kwargs["state_in"] = qkv_state_in
                if self.soft_lcu:
                    extra_kwargs["lcu_eta"] = self.lcu_eta[h]
                probs_h = sim_fn(
                    amp_h, t, marginals,
                    self.local_ry[h], self.local_rz[h], self.local_ent[h],
                    self.cross_zz[h], self.cross_macro_ry[h],
                    self.reup_ry[h], self.reup_time[h],
                    self.reup_data[h],
                    label_block_angles,
                    self.intra_reup_ry[h], self.intra_reup_time[h],
                    self.intra_reup_data[h],
                    self.post_ry1[h], self.post_rz1[h],
                    self.post_ry2[h],
                    nb=cfg.n_swin_blocks, nll=cfg.n_local_layers,
                    nq=cfg.n_qubits, intra_reup=cfg.intra_reup,
                    macro_n=self._macro_n, micro_n=self._micro_n,
                    macro_s=self._macro_s, micro_s=self._micro_s,
                    me_n=self._me_n, me_s=self._me_s,
                    ce_n=self._ce_n, ce_s=self._ce_s,
                    residual_logits=self.residual_logits if (self.residual_mixing and not cfg.lcu_residual) else None,
                    lcu_residual=cfg.lcu_residual,
                    **extra_kwargs,
                )
                if _compiled_qrf_sim[1] is not None:
                    import time as _time
                    print(f"  First compiled call complete — total compile time: "
                          f"{_time.perf_counter()-_compiled_qrf_sim[1]:.1f}s")
                    _compiled_qrf_sim[1] = None
            if qkv_score is not None:
                amp_out_h = torch.sqrt(probs_h.clamp(min=cfg.eps))
                amp_out_h = amp_out_h / amp_out_h.norm(dim=1, keepdim=True).clamp(min=cfg.eps)
                s = qkv_score.unsqueeze(1).to(dt)
                amp_combined = (1 - s) * amp + s * amp_out_h
                amp_combined = amp_combined / amp_combined.norm(dim=1, keepdim=True).clamp(min=cfg.eps)
                probs_h = amp_combined * amp_combined
            head_outputs.append(probs_h)

        combined = sum(head_outputs) / cfg.n_heads

        if self.hg_perm is not None:
            combined = combined[:, self.hg_perm]

        return combined

    def forward_state(self, x_t, t, labels, state_in=None):
        cfg = self.cfg
        dt = self.marginal_masks.dtype
        t = t.to(dt)
        if not cfg.use_time_input:
            t = torch.zeros_like(t)

        if state_in is not None:
            re_in, im_in = state_in
            B = re_in.shape[0]
            probs_in = (re_in * re_in + im_in * im_in).reshape(B, -1).to(dt)
            if self.hg_inv_perm is not None:
                probs_in = probs_in[:, self.hg_inv_perm]
            amp = None
            marginals = ((probs_in @ self.marginal_masks.T) - 0.5).T.contiguous()
        else:
            x_t = x_t.to(dt)
            if self.hg_inv_perm is not None:
                x_t = x_t[:, self.hg_inv_perm]
            amp = torch.sqrt(torch.clamp(x_t, min=cfg.eps))
            amp = amp / (amp.norm(dim=1, keepdim=True) + cfg.eps)
            marginals = ((x_t @ self.marginal_masks.T) - 0.5).T.contiguous()
            label_angles = self.label_embed[labels]
            amp = _apply_label_rotations(amp, label_angles, cfg.n_qubits)
            amp = amp.contiguous()

        label_block_angles = (
            cfg.label_reup_scale * self.label_reup[labels].permute(1, 2, 0)
        ).contiguous()

        head_states = []
        for h in range(cfg.n_heads):
            amp_h = amp
            if amp is not None and self.block_encode_proj:
                amp_h = apply_block_encoding_real(
                    amp,
                    self.block_ry_layers[h],
                    self.block_cry_anchor[h],
                    self.block_cry_data[h],
                    self.block_ry_final[h],
                    cfg.n_qubits,
                    eps=cfg.eps,
                )
            re, im = simulate_qrf_circuit_real(
                amp_h, t, marginals,
                self.local_ry[h], self.local_rz[h], self.local_ent[h],
                self.cross_zz[h], self.cross_macro_ry[h],
                self.reup_ry[h], self.reup_time[h], self.reup_data[h],
                label_block_angles,
                self.intra_reup_ry[h], self.intra_reup_time[h],
                self.intra_reup_data[h],
                self.post_ry1[h], self.post_rz1[h],
                self.post_ry2[h],
                nb=cfg.n_swin_blocks, nll=cfg.n_local_layers,
                nq=cfg.n_qubits, intra_reup=cfg.intra_reup,
                macro_n=self._macro_n, micro_n=self._micro_n,
                macro_s=self._macro_s, micro_s=self._micro_s,
                me_n=self._me_n, me_s=self._me_s,
                ce_n=self._ce_n, ce_s=self._ce_s,
                residual_logits=self.residual_logits if self.residual_mixing else None,
                state_in=state_in,
                return_probs=False,
            )
            head_states.append((re, im))

        if cfg.n_heads == 1:
            return head_states[0]
        probs_avg = sum(
            (re * re + im * im).reshape(re.shape[0], -1)
            for re, im in head_states
        ) / cfg.n_heads
        amp_avg = torch.sqrt(torch.clamp(probs_avg, min=cfg.eps))
        amp_avg = amp_avg / (amp_avg.norm(dim=1, keepdim=True) + cfg.eps)
        return _amp_to_state_real(amp_avg, cfg.n_qubits)


class SequentialQuantumVelocityNet(nn.Module):
    def __init__(self, cfg: RFConfig, n_stages: int = 2):
        super().__init__()
        self.n_stages = n_stages
        self.cfg = cfg
        self.stage_cond = cfg.stage_cond

        self.stages = nn.ModuleList()
        for _ in range(n_stages):
            stage = QuantumVelocityNet(cfg)
            self.stages.append(stage)

        if self.stage_cond:
            self.stage_label_embed = nn.Parameter(
                0.01 * torch.randn(n_stages, cfg.n_qubits, dtype=cfg.torch_dtype))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        if cfg.coherent_stages and self.n_stages > 1:
            state = self.stages[0].forward_state(x_t, t, labels, state_in=None)
            for stage in self.stages[1:]:
                state = stage.forward_state(x_t=None, t=t, labels=labels, state_in=state)
            re, im = state
            probs = (re * re + im * im).reshape(re.shape[0], -1)
            if self.stages[0].hg_perm is not None:
                probs = probs[:, self.stages[0].hg_perm]
            return probs.contiguous()

        x = x_t
        for k, stage in enumerate(self.stages):
            if self.stage_cond:
                progress = (k + 1) / self.n_stages
                offset = progress * self.stage_label_embed[k].unsqueeze(0)
                x = stage(x, t, labels, label_embed_offset=offset).contiguous()
            else:
                x = stage(x, t, labels).contiguous()
        return x
