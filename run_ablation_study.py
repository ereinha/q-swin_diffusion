import json
import time

import torch
import numpy as np

try:
    import torch._inductor.config as _inductor_cfg
    _inductor_cfg.cpp_cache_precompile_headers = False
except Exception:
    pass

import qrf_model as _qrf_model
from qrf_model import RFConfig, prob_to_sphere
from qrf_train import train_mnist_flow, generate_samples
from qrf_eval import evaluate_model


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30

BASELINE = dict(
    data_source="mnist",
    img_size=8,
    n_qubits=6,
    n_swin_blocks=6,
    n_local_layers=3,
    n_stages=5,
    n_heads=1,
    lr=5e-3,
    batch_size=64,
    epochs=EPOCHS,
    warmup_epochs=0,
    flow_space="sphere",
    loss_type="hellinger",
    label_reup_scale=1.0,
    intra_reup=True,
    optimizer="riemannian_adam",
    adam_eps=1e-8,
    weight_decay=0.0,
    schedule="cosine",
    pixel_order="gray2d_aligned",
    marginal_weight=1.0,
    focal_gamma=1.0,
    grad_clip=5.0,
    stage_cond=False,
    residual_mixing=True,
    lcu_residual=True,
    soft_lcu=True,
    soft_lcu_init=0.5,
    dtype="float32",
    use_jacobi=True,
    compile_sim=True,
    noise_type="slerp",
    use_pairwise_reup=True,
    use_rowcol_reup=False,
    prediction_mode="tangent",
    n_flow_steps=50,
    use_time_input=False,
    use_ot_pairing=True,
    diversity_weight=0.01,
    qkv_attention=True,
    qkv_layers=3,
)


GROUP_A = [
    ("baseline",                       {}),
    ("no focal gamma",                 {"focal_gamma": 0.0}),
    ("small focal gamma",              {"focal_gamma": 0.5}),
    ("large focal gamma",              {"focal_gamma": 2.0}),
    ("no marginal loss",               {"marginal_weight": 0.0}),
    ("small marginal loss",            {"marginal_weight": 0.1}),
    ("large marginal loss",            {"marginal_weight": 2.0}),
    ("plain adam",                     {"optimizer": "adam"}),
    ("warmup-only schedule",           {"schedule": "warmup", "warmup_epochs": 5}),
    ("no label reup",                  {"label_reup_scale": 0.0}),
    ("single stage (n=1)",             {"n_stages": 1}),
    ("3 stages (n=3)",                 {"n_stages": 3}),
    ("no diversity loss",              {"diversity_weight": 0.0}),
    ("higher diversity (w=0.04)",      {"diversity_weight": 0.04}),
    ("no ot pairing",                  {"use_ot_pairing": False}),
    ("pixel_order=native",             {"pixel_order": "native"}),
    ("with time input",                {"use_time_input": True}),
    ("x1 prediction mode",             {"prediction_mode": "x1", "use_time_input": True}),
]

GROUP_B = [
    ("fixed lcu (eta=pi/2)",           {"soft_lcu": False}),
]
GROUP_B2 = [
    ("no lcu residual",                {"lcu_residual": False, "soft_lcu": False}),
]
GROUP_B3 = [
    ("no residual mixing",             {"residual_mixing": False, "lcu_residual": True, "soft_lcu": False}),
]
GROUP_C = [
    ("no intra reup",                  {"intra_reup": False}),
]
GROUP_D = [
    ("no pairwise reup",               {"use_pairwise_reup": False}),
]
GROUP_E = [
    ("with rowcol reup",               {"use_rowcol_reup": True}),
]

GROUP_F = [
    ("fewer swin blocks (5)",          {"n_swin_blocks": 5}),
]

GROUP_G = [
    ("fewer local layers (2)",         {"n_local_layers": 2}),
]

GROUP_H = [
    ("fewer qkv layers (2)",            {"qkv_layers": 2}),
]

GROUP_I = [
    ("no qkv attention",               {"qkv_attention": False}),
]


def clear_compile_cache():
    _qrf_model._compiled_qrf_sim[0] = None
    _qrf_model._compiled_qrf_sim[1] = None


def compute_diversity(model, cfg, ink_stats, device, n_per_class=10, n_steps=100):
    dt = cfg.torch_dtype
    all_div = []
    for digit in range(10):
        imgs = generate_samples(model, cfg, num_samples=n_per_class,
                                labels=[digit] * n_per_class, n_steps=n_steps,
                                ink_stats=ink_stats, integrator="geodesic")
        probs = imgs.reshape(n_per_class, -1).to(dtype=dt, device="cpu")
        probs = probs / (probs.sum(1, keepdim=True) + cfg.eps)
        s = prob_to_sphere(probs, cfg.eps)
        gram = s @ s.T
        mask = ~torch.eye(n_per_class, dtype=torch.bool)
        angles = torch.acos(gram[mask].clamp(-1 + 1e-8, 1 - 1e-8))
        all_div.append(angles.mean().item())
    return float(np.mean(all_div))


def run_ablation(label, overrides, seed=SEED):
    params = {**BASELINE, **overrides}
    cfg = RFConfig(**params)

    t0 = time.time()
    model, cfg, stats, ink_stats = train_mnist_flow(cfg, seed=seed, device=DEVICE)
    train_time = time.time() - t0

    best_val = min(stats["val_loss"])
    final_val = stats["val_loss"][-1]
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Evaluating (FID, diversity, cond_acc)...")
    import matplotlib
    matplotlib.use('Agg')
    metrics = evaluate_model(model, cfg, num_gen=500, num_real=500,
                             ink_stats=ink_stats, n_flow_steps=100)
    diversity = compute_diversity(model, cfg, ink_stats, DEVICE)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        label=label,
        overrides=overrides,
        n_params=n_params,
        best_val=best_val,
        final_val=final_val,
        fid=metrics["fid"],
        cond_acc=metrics["cond_accuracy"],
        mmd_rbf=metrics["mmd_rbf"],
        mean_l2=metrics["mean_l2"],
        diversity=diversity,
        val_losses=stats["val_loss"],
        time=train_time,
    )


def print_results(results):
    baseline_val = results[0]["best_val"]

    print(f"\nABLATION RESULTS")
    print(f"  {'Condition':<32}  {'params':>6}  {'best_val':>9}  {'FID':>7}  "
          f"{'cond_acc':>8}  {'divers':>7}  {'MMD':>9}  {'L2':>7}  {'time':>6}")
    for r in results:
        print(f"  {r['label']:<32}  {r['n_params']:>6}  {r['best_val']:>9.6f}  "
              f"{r['fid']:>7.1f}  {r['cond_acc']:>8.4f}  {r['diversity']:>7.4f}  "
              f"{r['mmd_rbf']:>9.6f}  {r['mean_l2']:>7.4f}  {r['time']:>5.0f}s")

    print(f"\nRanked by FID (lower = better):")
    sorted_fid = sorted(results, key=lambda r: r["fid"])
    for i, r in enumerate(sorted_fid, 1):
        print(f"  {i:>2}. {r['label']:<32}  FID={r['fid']:>7.1f}  val={r['best_val']:.6f}  "
              f"div={r['diversity']:.4f}  cond={r['cond_acc']:.4f}")

    print(f"\nRanked by val loss impact (most harmful removal first):")
    sorted_r = sorted(results[1:], key=lambda r: r["best_val"] - baseline_val, reverse=True)
    for i, r in enumerate(sorted_r, 1):
        delta = r["best_val"] - baseline_val
        pct = delta / max(baseline_val, 1e-8) * 100
        print(f"  {i:>2}. {r['label']:<32}  val_delta={delta:>+.6f}  ({pct:>+.1f}%)  "
              f"FID={r['fid']:.1f}")

    show_epochs = [e for e in range(EPOCHS) if e % 5 == 4 or e == 0]
    print(f"\nVal loss at selected epochs:")
    hdr = f"  {'Condition':<32}  " + "  ".join(f"e{e+1:02d}" for e in show_epochs)
    print(hdr)
    for r in results:
        vals = [r["val_losses"][e] for e in show_epochs if e < len(r["val_losses"])]
        row = f"  {r['label']:<32}  " + "  ".join(f"{v:.4f}" for v in vals)
        print(row)


CHECKPOINT_FILE = "ablation_checkpoint.json"


def load_checkpoint():
    import os
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        print(f"Resuming from checkpoint ({len(data['results'])} conditions completed)")
        return data["results"], set(data["completed_labels"])
    return [], set()


def save_checkpoint(results):
    data = {
        "completed_labels": [r["label"] for r in results],
        "results": results,
    }
    import os
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CHECKPOINT_FILE)
    print(f"  [checkpoint saved: {len(results)} conditions -> {os.path.abspath(CHECKPOINT_FILE)}]")


def main():
    import matplotlib
    matplotlib.use('Agg')

    groups = [
        ("8x8 MNIST baseline (param-only ablations)",    GROUP_A),
        ("fewer local layers",                           GROUP_G),
        ("fewer qkv layers",                             GROUP_H),
        ("fewer swin blocks",                            GROUP_F),
        ("fixed LCU (eta=pi/2)",                         GROUP_B),
        ("no lcu residual",                              GROUP_B2),
        ("no intra reup",                                GROUP_C),
        ("no pairwise reup",                             GROUP_D),
        ("no qkv attention",                             GROUP_I),
        ("no residual mixing",                           GROUP_B3),
        ("with rowcol reup",                             GROUP_E),
    ]
    total = sum(len(g) for _, g in groups)
    print(f"Ablation study: {total} conditions x {EPOCHS} epochs")
    print(f"Baseline: 8x8 MNIST, 6q, swin_blocks=4, local_layers=3, n_stages=5, "
          f"slerp+OT, tangent mode, diversity_w=0.01")
    print(f"Device: {DEVICE}")

    results, completed = load_checkpoint()

    for group_label, conditions in groups:
        remaining = [c for c in conditions if c[0] not in completed]
        if not remaining:
            continue
        print(f"\n{group_label} ({len(remaining)} remaining)")
        clear_compile_cache()

        for label, overrides in conditions:
            if label in completed:
                continue
            print(f"\n  {label}" + (f"  {overrides}" if overrides else ""))
            r = run_ablation(label, overrides)
            results.append(r)
            completed.add(label)
            save_checkpoint(results)
            print(f"  val={r['best_val']:.6f}  FID={r['fid']:.1f}  cond={r['cond_acc']:.4f}  "
                  f"div={r['diversity']:.4f}  params={r['n_params']}  time={r['time']:.0f}s")

    print_results(results)

    with open("ablation_study_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFinal results saved to ablation_study_results.json")


if __name__ == "__main__":
    main()
