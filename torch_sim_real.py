import torch


def _bc(val, ndim):
    if val.dim() == 0:
        return val
    return val.reshape(-1, *([1] * ndim))


def apply_ry(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    c = _bc(torch.cos(theta / 2), nq - 1)
    s = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([c * re0 - s * re1, s * re0 + c * re1], dim=-1)
    new_im = torch.stack([c * im0 - s * im1, s * im0 + c * im1], dim=-1)
    return new_re.movedim(-1, qubit + 1), new_im.movedim(-1, qubit + 1)


def apply_rz(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    ct = _bc(torch.cos(theta / 2), nq - 1)
    st = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re0 = ct * re0 + st * im0
    new_im0 = ct * im0 - st * re0
    new_re1 = ct * re1 - st * im1
    new_im1 = ct * im1 + st * re1
    return (torch.stack([new_re0, new_re1], dim=-1).movedim(-1, qubit + 1),
            torch.stack([new_im0, new_im1], dim=-1).movedim(-1, qubit + 1))


def apply_rx(re, im, theta, qubit, nq):
    re = re.movedim(qubit + 1, -1)
    im = im.movedim(qubit + 1, -1)
    c = _bc(torch.cos(theta / 2), nq - 1)
    s = _bc(torch.sin(theta / 2), nq - 1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re0 = c * re0 + s * im1
    new_im0 = c * im0 - s * re1
    new_re1 = s * im0 + c * re1
    new_im1 = -s * re0 + c * im1
    return (torch.stack([new_re0, new_re1], dim=-1).movedim(-1, qubit + 1),
            torch.stack([new_im0, new_im1], dim=-1).movedim(-1, qubit + 1))


def apply_izz(re, im, theta, q0, q1, nq):
    z = torch.tensor([1.0, -1.0], dtype=re.dtype, device=re.device)
    s0 = [1] * (nq + 1); s0[q0 + 1] = 2
    s1 = [1] * (nq + 1); s1[q1 + 1] = 2
    zz = z.reshape(s0) * z.reshape(s1)
    angle = _bc(theta / 2, nq) * zz
    ct = torch.cos(angle)
    st = torch.sin(angle)
    new_re = ct * re + st * im
    new_im = ct * im - st * re
    return new_re, new_im


def _two_qubit_perm(control, target, nq):
    target_axes = [control + 1, target + 1]
    others = [a for a in range(1, nq + 1) if a not in target_axes]
    perm = [0] + others + target_axes
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return perm, inv


def apply_cry(re, im, theta, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_c1 = torch.stack([c * re_t0 - s * re_t1, s * re_t0 + c * re_t1], dim=-1)
    new_im_c1 = torch.stack([c * im_t0 - s * im_t1, s * im_t0 + c * im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_x(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    new_re = torch.stack([re[..., 1], re[..., 0]], dim=-1)
    new_im = torch.stack([im[..., 1], im[..., 0]], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_y(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([im1, -im0], dim=-1)
    new_im = torch.stack([-re1, re0], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_z(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    new_re = torch.stack([re[..., 0], -re[..., 1]], dim=-1)
    new_im = torch.stack([im[..., 0], -im[..., 1]], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_h(re, im, q, nq):
    inv_sqrt2 = 0.7071067811865475
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    re0, re1 = re[..., 0], re[..., 1]
    im0, im1 = im[..., 0], im[..., 1]
    new_re = torch.stack([(re0 + re1) * inv_sqrt2, (re0 - re1) * inv_sqrt2], dim=-1)
    new_im = torch.stack([(im0 + im1) * inv_sqrt2, (im0 - im1) * inv_sqrt2], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_s(re, im, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    new_re = torch.stack([re[..., 0], -im[..., 1]], dim=-1)
    new_im = torch.stack([im[..., 0], re[..., 1]], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_t(re, im, q, nq):
    inv_sqrt2 = 0.7071067811865475
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    re1, im1 = re[..., 1], im[..., 1]
    new_re1 = (re1 - im1) * inv_sqrt2
    new_im1 = (re1 + im1) * inv_sqrt2
    new_re = torch.stack([re[..., 0], new_re1], dim=-1)
    new_im = torch.stack([im[..., 0], new_im1], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_phase_shift(re, im, phi, q, nq):
    re = re.movedim(q + 1, -1); im = im.movedim(q + 1, -1)
    c = _bc(torch.cos(phi), nq - 1)
    s = _bc(torch.sin(phi), nq - 1)
    re1, im1 = re[..., 1], im[..., 1]
    new_re1 = c * re1 - s * im1
    new_im1 = c * im1 + s * re1
    new_re = torch.stack([re[..., 0], new_re1], dim=-1)
    new_im = torch.stack([im[..., 0], new_im1], dim=-1)
    return new_re.movedim(-1, q + 1), new_im.movedim(-1, q + 1)


def apply_rot(re, im, phi, theta, omega, q, nq):
    re, im = apply_rz(re, im, phi, q, nq)
    re, im = apply_ry(re, im, theta, q, nq)
    re, im = apply_rz(re, im, omega, q, nq)
    return re, im


def apply_cnot(re, im, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    new_re_c1 = torch.stack([re_c1[..., 1], re_c1[..., 0]], dim=-1)
    new_im_c1 = torch.stack([im_c1[..., 1], im_c1[..., 0]], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


apply_cx = apply_cnot


def apply_cy(re, im, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_c1 = torch.stack([im_t1, -im_t0], dim=-1)
    new_im_c1 = torch.stack([-re_t1, re_t0], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_cz(re, im, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    new_re_c1 = torch.stack([re_c1[..., 0], -re_c1[..., 1]], dim=-1)
    new_im_c1 = torch.stack([im_c1[..., 0], -im_c1[..., 1]], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_swap(re, im, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    out_re = re.transpose(-1, -2)
    out_im = im.transpose(-1, -2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_iswap(re, im, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    re_00 = re[..., 0, 0]; re_01 = re[..., 0, 1]
    re_10 = re[..., 1, 0]; re_11 = re[..., 1, 1]
    im_00 = im[..., 0, 0]; im_01 = im[..., 0, 1]
    im_10 = im[..., 1, 0]; im_11 = im[..., 1, 1]
    new_re_10, new_im_10 = -im_01, re_01
    new_re_01, new_im_01 = -im_10, re_10
    row0_re = torch.stack([re_00, new_re_01], dim=-1)
    row1_re = torch.stack([new_re_10, re_11], dim=-1)
    row0_im = torch.stack([im_00, new_im_01], dim=-1)
    row1_im = torch.stack([new_im_10, im_11], dim=-1)
    out_re = torch.stack([row0_re, row1_re], dim=-2)
    out_im = torch.stack([row0_im, row1_im], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_crx(re, im, theta, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_t0 = c * re_t0 + s * im_t1
    new_im_t0 = c * im_t0 - s * re_t1
    new_re_t1 = s * im_t0 + c * re_t1
    new_im_t1 = -s * re_t0 + c * im_t1
    new_re_c1 = torch.stack([new_re_t0, new_re_t1], dim=-1)
    new_im_c1 = torch.stack([new_im_t0, new_im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_crz(re, im, theta, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    ct = _bc(torch.cos(theta / 2), nq - 2)
    st = _bc(torch.sin(theta / 2), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_t0 = ct * re_t0 + st * im_t0
    new_im_t0 = ct * im_t0 - st * re_t0
    new_re_t1 = ct * re_t1 - st * im_t1
    new_im_t1 = ct * im_t1 + st * re_t1
    new_re_c1 = torch.stack([new_re_t0, new_re_t1], dim=-1)
    new_im_c1 = torch.stack([new_im_t0, new_im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_cphase(re, im, phi, control, target, nq):
    perm, inv = _two_qubit_perm(control, target, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(phi), nq - 2)
    s = _bc(torch.sin(phi), nq - 2)
    re_c0 = re[..., 0, :]; im_c0 = im[..., 0, :]
    re_c1 = re[..., 1, :]; im_c1 = im[..., 1, :]
    re_t0, re_t1 = re_c1[..., 0], re_c1[..., 1]
    im_t0, im_t1 = im_c1[..., 0], im_c1[..., 1]
    new_re_t1 = c * re_t1 - s * im_t1
    new_im_t1 = c * im_t1 + s * re_t1
    new_re_c1 = torch.stack([re_t0, new_re_t1], dim=-1)
    new_im_c1 = torch.stack([im_t0, new_im_t1], dim=-1)
    out_re = torch.stack([re_c0, new_re_c1], dim=-2)
    out_im = torch.stack([im_c0, new_im_c1], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_ixx(re, im, theta, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_00 = re[..., 0, 0]; re_01 = re[..., 0, 1]
    re_10 = re[..., 1, 0]; re_11 = re[..., 1, 1]
    im_00 = im[..., 0, 0]; im_01 = im[..., 0, 1]
    im_10 = im[..., 1, 0]; im_11 = im[..., 1, 1]
    new_re_00 = c * re_00 + s * im_11; new_im_00 = c * im_00 - s * re_11
    new_re_11 = c * re_11 + s * im_00; new_im_11 = c * im_11 - s * re_00
    new_re_01 = c * re_01 + s * im_10; new_im_01 = c * im_01 - s * re_10
    new_re_10 = c * re_10 + s * im_01; new_im_10 = c * im_10 - s * re_01
    row0_re = torch.stack([new_re_00, new_re_01], dim=-1)
    row1_re = torch.stack([new_re_10, new_re_11], dim=-1)
    row0_im = torch.stack([new_im_00, new_im_01], dim=-1)
    row1_im = torch.stack([new_im_10, new_im_11], dim=-1)
    out_re = torch.stack([row0_re, row1_re], dim=-2)
    out_im = torch.stack([row0_im, row1_im], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def apply_iyy(re, im, theta, q0, q1, nq):
    perm, inv = _two_qubit_perm(q0, q1, nq)
    re = re.permute(perm); im = im.permute(perm)
    c = _bc(torch.cos(theta / 2), nq - 2)
    s = _bc(torch.sin(theta / 2), nq - 2)
    re_00 = re[..., 0, 0]; re_01 = re[..., 0, 1]
    re_10 = re[..., 1, 0]; re_11 = re[..., 1, 1]
    im_00 = im[..., 0, 0]; im_01 = im[..., 0, 1]
    im_10 = im[..., 1, 0]; im_11 = im[..., 1, 1]
    new_re_00 = c * re_00 - s * im_11; new_im_00 = c * im_00 + s * re_11
    new_re_11 = c * re_11 - s * im_00; new_im_11 = c * im_11 + s * re_00
    new_re_01 = c * re_01 + s * im_10; new_im_01 = c * im_01 - s * re_10
    new_re_10 = c * re_10 + s * im_01; new_im_10 = c * im_10 - s * re_01
    row0_re = torch.stack([new_re_00, new_re_01], dim=-1)
    row1_re = torch.stack([new_re_10, new_re_11], dim=-1)
    row0_im = torch.stack([new_im_00, new_im_01], dim=-1)
    row1_im = torch.stack([new_im_10, new_im_11], dim=-1)
    out_re = torch.stack([row0_re, row1_re], dim=-2)
    out_im = torch.stack([row0_im, row1_im], dim=-2)
    return out_re.permute(inv), out_im.permute(inv)


def _move_three_to_end(re, im, q1, q2, q3, nq):
    target_axes = [q1 + 1, q2 + 1, q3 + 1]
    others = [a for a in range(1, nq + 1) if a not in target_axes]
    perm = [0] + others + target_axes
    return re.permute(perm), im.permute(perm), perm


def _invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def apply_toffoli(re, im, c1, c2, target, nq):
    re_p, im_p, perm = _move_three_to_end(re, im, c1, c2, target, nq)
    re_00 = re_p[..., 0, 0, :]; im_00 = im_p[..., 0, 0, :]
    re_01 = re_p[..., 0, 1, :]; im_01 = im_p[..., 0, 1, :]
    re_10 = re_p[..., 1, 0, :]; im_10 = im_p[..., 1, 0, :]
    re_11 = re_p[..., 1, 1, :]; im_11 = im_p[..., 1, 1, :]
    re_11_new = re_11.flip(-1)
    im_11_new = im_11.flip(-1)
    row0 = torch.stack([re_00, re_01], dim=-2)
    row1 = torch.stack([re_10, re_11_new], dim=-2)
    new_re = torch.stack([row0, row1], dim=-3)
    row0i = torch.stack([im_00, im_01], dim=-2)
    row1i = torch.stack([im_10, im_11_new], dim=-2)
    new_im = torch.stack([row0i, row1i], dim=-3)
    return new_re.permute(_invert_perm(perm)), new_im.permute(_invert_perm(perm))


apply_ccx = apply_toffoli


def apply_cswap(re, im, control, target1, target2, nq):
    re_p, im_p, perm = _move_three_to_end(re, im, control, target1, target2, nq)
    re_c0 = re_p[..., 0, :, :]; im_c0 = im_p[..., 0, :, :]
    re_c1 = re_p[..., 1, :, :]; im_c1 = im_p[..., 1, :, :]
    new_re_c1 = re_c1.transpose(-1, -2)
    new_im_c1 = im_c1.transpose(-1, -2)
    new_re = torch.stack([re_c0, new_re_c1], dim=-3)
    new_im = torch.stack([im_c0, new_im_c1], dim=-3)
    return new_re.permute(_invert_perm(perm)), new_im.permute(_invert_perm(perm))


def _amp_to_state_real(amp, nq):
    B = amp.shape[0]
    norm = amp.norm(dim=1, keepdim=True).clamp(min=1e-12)
    normed = (amp / norm).reshape(B, *([2] * nq))
    return normed, torch.zeros_like(normed)


def apply_block_encoding_complex(amp, rot_layers, cry_anchor, cry_data, rot_final,
                                 nq, eps=1e-12):
    """Block-encoded projection with Rot(φ,θ,ω)-gate ansatz, returning complex
    post-selected state on the data register.

    Each per-qubit single-qubit gate is a full SU(2) rotation Rot(φ,θ,ω) =
    Rz(ω)·Ry(θ)·Rz(φ), giving 3 DOFs per qubit. Entanglers stay CRY (anchor
    ancilla→data and brick data_q→data_{q+1}). The post-selected operator
    W(θ) = ⟨0_a|U(θ)|0_a⟩ is a complex contraction (sub-norm 1).

    Layout: data qubits 0..nq-1, ancilla = qubit nq.
    Convention: ancilla post-selected on |0⟩ corresponds to a measurement
    that is reset before any subsequent ancilla use elsewhere.

    Args:
      amp: (B, 2^nq) real input amplitude vector.
      rot_layers: (L, nq+1, 3) per-qubit per-layer Rot angles (φ, θ, ω).
      cry_anchor: (L, nq) anchor CRY angles.
      cry_data: (L, nq-1) brick CRY angles.
      rot_final: (nq+1, 3) final per-qubit Rot angles.

    Returns:
      (re_post, im_post): each shape (B, *([2]*nq)) — post-selected on
      ancilla=|0⟩ and ℓ₂-renormalized.
    """
    B = amp.shape[0]
    nq_total = nq + 1

    norm_in = amp.norm(dim=1, keepdim=True).clamp(min=eps)
    amp_n = amp / norm_in

    re_data = amp_n.reshape(B, *([2] * nq))
    zero = torch.zeros_like(re_data)
    re = torch.stack([re_data, zero], dim=-1)
    im = torch.zeros_like(re)

    L = rot_layers.shape[0]
    for layer in range(L):
        for q in range(nq_total):
            phi, theta, omega = rot_layers[layer, q, 0], rot_layers[layer, q, 1], rot_layers[layer, q, 2]
            re, im = apply_rot(re, im, phi, theta, omega, q, nq_total)
        for q in range(nq):
            re, im = apply_cry(re, im, cry_anchor[layer, q], nq, q, nq_total)
        for q in range(nq - 1):
            re, im = apply_cry(re, im, cry_data[layer, q], q, q + 1, nq_total)

    for q in range(nq_total):
        phi, theta, omega = rot_final[q, 0], rot_final[q, 1], rot_final[q, 2]
        re, im = apply_rot(re, im, phi, theta, omega, q, nq_total)

    re_post = re[..., 0]
    im_post = im[..., 0]

    norm_sq = (re_post * re_post + im_post * im_post).reshape(B, -1).sum(dim=1)
    norm = torch.sqrt(norm_sq.clamp(min=eps * eps)).reshape(-1, *([1] * nq))
    return re_post / norm, im_post / norm


def hadamard_test_overlap(re_Q, im_Q, re_K, im_K, nq):
    """Re⟨amp_Q | amp_K⟩ via gate-level Hadamard test.

    Procedure:
      1. Joint state (1/√2)(|0⟩_h |amp_Q⟩ + |1⟩_h |amp_K⟩) on (n_q+1) qubits.
         The h-ancilla is freshly initialized (the block-encoding ancillas of
         W_Q and W_K were already post-selected to produce |amp_Q⟩, |amp_K⟩;
         that is what corresponds to "reset" in a hardware circuit).
      2. Apply H on h.
      3. P(|0⟩_h) = (1 + Re⟨amp_Q|amp_K⟩) / 2  ⇒  Re⟨Q|K⟩ = 2·P(|0⟩_h) − 1.

    Args:
      re_Q, im_Q, re_K, im_K: each shape (B, *([2]*nq)) — normalized states.

    Returns:
      Tensor of shape (B,): Re⟨amp_Q | amp_K⟩.
    """
    inv_sqrt2 = 0.7071067811865475
    re_joint = torch.stack([re_Q, re_K], dim=1) * inv_sqrt2
    im_joint = torch.stack([im_Q, im_K], dim=1) * inv_sqrt2

    nq_total = nq + 1
    re_joint, im_joint = apply_h(re_joint, im_joint, q=0, nq=nq_total)

    re_h0 = re_joint[:, 0]
    im_h0 = im_joint[:, 0]
    p0 = (re_h0 * re_h0 + im_h0 * im_h0).reshape(re_h0.shape[0], -1).sum(dim=1)
    return 2.0 * p0 - 1.0


def apply_block_encoding_real(amp, ry_layers, cry_anchor, cry_data, ry_final, nq,
                              eps=1e-12):
    """Block-encoded projection via a parameterized (nq+1)-qubit unitary.

    Construction:
      |Ψ_in⟩ = |0⟩_a ⊗ |amp⟩
      |Ψ_out⟩ = U(θ) |Ψ_in⟩
      output_amp = ⟨0_a|Ψ_out⟩, renormalized to unit ℓ₂ norm.

    The post-selected branch implements ⟨0_a|U(θ)|0_a⟩ = W(θ) acting on the
    data register — a learnable block-encoded operator. Since U(θ) uses only
    Ry / CRY gates (real-valued), the post-selected state stays real.

    Layout: data qubits 0..nq-1, ancilla = qubit nq.
    Per layer: Ry on each of (nq+1) qubits, anchor CRY ancilla→data_q for each
    data qubit, brick-pattern CRY data_q→data_{q+1}.

    Args:
      amp: (B, 2^nq) real input amplitude vector.
      ry_layers: (L, nq+1) Ry angles per layer.
      cry_anchor: (L, nq) anchor CRY angles (ancilla → data_q).
      cry_data: (L, nq-1) brick-pattern CRY angles (data_q → data_{q+1}).
      ry_final: (nq+1,) final Ry angles.
      nq: number of data qubits.

    Returns:
      (B, 2^nq) real amplitude post-selected on ancilla=|0⟩, ℓ₂-normalized.
    """
    B = amp.shape[0]
    nq_total = nq + 1

    norm_in = amp.norm(dim=1, keepdim=True).clamp(min=eps)
    amp_n = amp / norm_in

    re_data = amp_n.reshape(B, *([2] * nq))
    zero = torch.zeros_like(re_data)
    re = torch.stack([re_data, zero], dim=-1)
    im = torch.zeros_like(re)

    L = ry_layers.shape[0]
    for layer in range(L):
        for q in range(nq_total):
            re, im = apply_ry(re, im, ry_layers[layer, q], q, nq_total)
        for q in range(nq):
            re, im = apply_cry(re, im, cry_anchor[layer, q], nq, q, nq_total)
        for q in range(nq - 1):
            re, im = apply_cry(re, im, cry_data[layer, q], q, q + 1, nq_total)

    for q in range(nq_total):
        re, im = apply_ry(re, im, ry_final[q], q, nq_total)

    re_post = re[..., 0]
    flat = re_post.reshape(B, -1)
    norm_out = flat.norm(dim=1, keepdim=True).clamp(min=eps)
    return flat / norm_out


def _state_to_probs_real(re, im):
    B = re.shape[0]
    return (re * re + im * im).reshape(B, -1)


def _renormalize_real(re, im, nq):
    B = re.shape[0]
    norm_sq = (re * re + im * im).reshape(B, -1).sum(dim=1)
    norm = torch.sqrt(norm_sq.clamp(min=1e-24)).reshape(-1, *([1] * nq))
    return re / norm, im / norm


def simulate_qrf_circuit_real(
    amp, t_val, marginals,
    local_ry, local_rz, local_ent,
    cross_zz, cross_macro_ry,
    reup_ry, reup_time, reup_data,
    label_angles,
    intra_reup_ry, intra_reup_time, intra_reup_data,
    post_ry1, post_rz1, post_ry2,
    nb, nll, nq, intra_reup,
    macro_n, micro_n, macro_s, micro_s,
    me_n, me_s, ce_n, ce_s,
    residual_logits=None,
    block_scale=None,
    state_in=None,
    return_probs=True,
    pair_marginals=None, pair_reup_rz=None, pair_reup_data=None, pair_list=None,
    row_marginals=None, rowcol_reup_ry=None, rowcol_reup_data=None,
    lcu_residual=False,
    lcu_eta=None,
):
    use_residual = residual_logits is not None or lcu_residual
    if state_in is not None:
        re, im = state_in
    else:
        re, im = _amp_to_state_real(amp, nq)

    for b in range(nb):
        if use_residual:
            re_before, im_before = re.clone(), im.clone()

        sc = block_scale[b] if block_scale is not None else None

        if b % 2 == 0:
            w_micro, micro_edges = micro_n, me_n
            w_macro, cross_edges = macro_n, ce_n
        else:
            w_micro, micro_edges = micro_s, me_s
            w_macro, cross_edges = macro_s, ce_s

        for ll in range(nll):
            for k, w in enumerate(w_micro):
                ry_val = local_ry[b, ll, k] * sc if sc is not None else local_ry[b, ll, k]
                rz_val = local_rz[b, ll, k] * sc if sc is not None else local_rz[b, ll, k]
                re, im = apply_ry(re, im, ry_val, w, nq)
                re, im = apply_rz(re, im, rz_val, w, nq)
            for k, (wa, wb) in enumerate(micro_edges):
                ent_val = local_ent[b, ll, k] * sc if sc is not None else local_ent[b, ll, k]
                re, im = apply_cry(re, im, ent_val, wa, wb, nq)
            if intra_reup and ll < nll - 1:
                for q in range(nq):
                    angle = (intra_reup_ry[b, ll, q]
                             + intra_reup_time[b, ll, q] * t_val
                             + intra_reup_data[b, ll, q] * marginals[q])
                    if sc is not None:
                        angle = angle * sc
                    re, im = apply_ry(re, im, angle, q, nq)

                if pair_marginals is not None and pair_reup_rz is not None:
                    for k, (qa, qb) in enumerate(pair_list):
                        angle = pair_reup_rz[b, ll, k] + pair_reup_data[b, ll, k] * pair_marginals[k]
                        re, im = apply_rz(re, im, angle, qa, nq)

                if row_marginals is not None and rowcol_reup_ry is not None:
                    for q in range(nq):
                        rc_idx = q % rowcol_reup_data.shape[3]
                        angle = rowcol_reup_ry[b, ll, q] + rowcol_reup_data[b, ll, q, rc_idx] * row_marginals[rc_idx]
                        re, im = apply_ry(re, im, angle, q, nq)

        for k, w in enumerate(w_macro):
            ry_val = cross_macro_ry[b, k] * sc if sc is not None else cross_macro_ry[b, k]
            re, im = apply_ry(re, im, ry_val, w, nq)
        for k, (wa, wb) in enumerate(cross_edges):
            zz_val = cross_zz[b, k] * sc if sc is not None else cross_zz[b, k]
            re, im = apply_cry(re, im, zz_val, wa, wb, nq)

        for q in range(nq):
            angle = (reup_ry[b, q] + reup_time[b, q] * t_val
                     + reup_data[b, q] * marginals[q])
            if sc is not None:
                angle = angle * sc
            re, im = apply_ry(re, im, angle, q, nq)
            lbl_val = label_angles[b, q] * sc if sc is not None else label_angles[b, q]
            re, im = apply_rx(re, im, lbl_val, q, nq)

        if use_residual:
            if lcu_residual:
                if lcu_eta is not None:
                    eta = lcu_eta[b]
                    c = torch.cos(eta / 2)
                    s = torch.sin(eta / 2)
                    re = c * re_before + s * re
                    im = c * im_before + s * im
                else:
                    re = re + re_before
                    im = im + im_before
                re, im = _renormalize_real(re, im, nq)
            else:
                gate = torch.sigmoid(residual_logits[b])
                re = gate * re + (1 - gate) * re_before
                im = gate * im + (1 - gate) * im_before
                re, im = _renormalize_real(re, im, nq)

    for q in range(nq):
        re, im = apply_ry(re, im, post_ry1[q], q, nq)
        re, im = apply_rz(re, im, post_rz1[q], q, nq)
    for q in range(nq):
        re, im = apply_ry(re, im, post_ry2[q], q, nq)

    if return_probs:
        return _state_to_probs_real(re, im)
    return re, im


def simulate_classify_circuit_real(
    amp, marginals,
    local_ry, local_rz, local_ent,
    cross_zz, cross_macro_ry,
    reup_ry, reup_data,
    intra_reup_ry, intra_reup_data,
    post_ry1, post_ry2,
    nb, nll, nq, intra_reup,
    macro_n, micro_n, macro_s, micro_s,
    me_n, me_s, ce_n, ce_s,
    residual_logits=None,
):
    re, im = _amp_to_state_real(amp, nq)

    for b in range(nb):
        if residual_logits is not None:
            re_before, im_before = re.clone(), im.clone()

        if b % 2 == 0:
            w_micro, micro_edges = micro_n, me_n
            w_macro, cross_edges = macro_n, ce_n
        else:
            w_micro, micro_edges = micro_s, me_s
            w_macro, cross_edges = macro_s, ce_s

        for ll in range(nll):
            for k, w in enumerate(w_micro):
                re, im = apply_ry(re, im, local_ry[b, ll, k], w, nq)
                re, im = apply_rz(re, im, local_rz[b, ll, k], w, nq)
            for k, (wa, wb) in enumerate(micro_edges):
                re, im = apply_cry(re, im, local_ent[b, ll, k], wa, wb, nq)
            if intra_reup and ll < nll - 1:
                for q in range(nq):
                    angle = (intra_reup_ry[b, ll, q]
                             + intra_reup_data[b, ll, q] * marginals[q])
                    re, im = apply_ry(re, im, angle, q, nq)

        for k, w in enumerate(w_macro):
            re, im = apply_ry(re, im, cross_macro_ry[b, k], w, nq)
        for k, (wa, wb) in enumerate(cross_edges):
            re, im = apply_cry(re, im, cross_zz[b, k], wa, wb, nq)

        for q in range(nq):
            angle = reup_ry[b, q] + reup_data[b, q] * marginals[q]
            re, im = apply_ry(re, im, angle, q, nq)

        if residual_logits is not None:
            gate = torch.sigmoid(residual_logits[b])
            re = gate * re + (1 - gate) * re_before
            im = gate * im + (1 - gate) * im_before
            re, im = _renormalize_real(re, im, nq)

    for q in range(nq):
        re, im = apply_ry(re, im, post_ry1[q], q, nq)
    for q in range(nq):
        re, im = apply_ry(re, im, post_ry2[q], q, nq)

    return _state_to_probs_real(re, im)


def simulate_toy_circuit_real(
    amp, t_val, marginals, joint_marg, axis_sign,
    local_ry, local_rz, local_ent,
    cross_zz, cross_macro_ry,
    reup_ry, reup_time, reup_data, reup_joint,
    reup_axis, intra_reup_axis,
    cond_block_angles, cond_rz_angles,
    intra_reup_ry, intra_reup_time, intra_reup_data,
    intra_reup_joint,
    intra_cond_rz_angles,
    post_ry1, post_rz1, post_ry2, post_rz2,
    post2_ry, post2_rz,
    nb, nll, nq, n_bits, intra_reup,
    macro_n, micro_n, macro_s, micro_s,
    me_n, me_s, ce_n, ce_s,
    ce_n_is_cross, ce_s_is_cross,
    joint_marg_flag, double_post_flag, cross_cry_flag, axis_sym_break,
    residual_logits=None,
):
    re, im = _amp_to_state_real(amp, nq)

    if axis_sym_break:
        pi_4 = torch.tensor(0.7854, dtype=re.dtype, device=re.device)
        for q in range(n_bits):
            re, im = apply_rz(re, im, pi_4, q, nq)

    for b in range(nb):
        if residual_logits is not None:
            re_before, im_before = re.clone(), im.clone()

        if b % 2 == 0:
            w_micro, micro_edges = micro_n, me_n
            w_macro, cross_edges = macro_n, ce_n
            ce_is_cross = ce_n_is_cross
        else:
            w_micro, micro_edges = micro_s, me_s
            w_macro, cross_edges = macro_s, ce_s
            ce_is_cross = ce_s_is_cross

        for ll in range(nll):
            for k, w in enumerate(w_micro):
                re, im = apply_ry(re, im, local_ry[b, ll, k], w, nq)
                re, im = apply_rz(re, im, local_rz[b, ll, k], w, nq)
            for k, (wa, wb) in enumerate(micro_edges):
                re, im = apply_izz(re, im, local_ent[b, ll, k], wa, wb, nq)
            if intra_reup and ll < nll - 1:
                for q in range(nq):
                    angle = (intra_reup_ry[b, ll, q]
                             + intra_reup_time[b, ll, q] * t_val
                             + intra_reup_data[b, ll, q] * marginals[q]
                             + intra_reup_axis[b, ll, q] * axis_sign[q])
                    if joint_marg_flag:
                        pair_idx = q if q < n_bits else q - n_bits
                        angle = angle + intra_reup_joint[b, ll, q] * joint_marg[pair_idx]
                    re, im = apply_ry(re, im, angle, q, nq)
                    re, im = apply_rz(re, im, intra_cond_rz_angles[b, ll, q], q, nq)

        for k, w in enumerate(w_macro):
            re, im = apply_ry(re, im, cross_macro_ry[b, k], w, nq)

        for k, (wa, wb) in enumerate(cross_edges):
            if cross_cry_flag and ce_is_cross[k]:
                re, im = apply_cry(re, im, cross_zz[b, k], wa, wb, nq)
            else:
                re, im = apply_izz(re, im, cross_zz[b, k], wa, wb, nq)

        for q in range(nq):
            angle = (reup_ry[b, q] + reup_time[b, q] * t_val
                     + reup_data[b, q] * marginals[q]
                     + reup_axis[b, q] * axis_sign[q])
            if joint_marg_flag:
                pair_idx = q if q < n_bits else q - n_bits
                angle = angle + reup_joint[b, q] * joint_marg[pair_idx]
            re, im = apply_ry(re, im, angle, q, nq)
            re, im = apply_rz(re, im, cond_rz_angles[b, q], q, nq)
            re, im = apply_rx(re, im, cond_block_angles[b, q], q, nq)

        if residual_logits is not None:
            gate = torch.sigmoid(residual_logits[b])
            re = gate * re + (1 - gate) * re_before
            im = gate * im + (1 - gate) * im_before
            re, im = _renormalize_real(re, im, nq)

    for q in range(nq):
        re, im = apply_ry(re, im, post_ry1[q], q, nq)
        re, im = apply_rz(re, im, post_rz1[q], q, nq)
    for q in range(nq):
        re, im = apply_ry(re, im, post_ry2[q], q, nq)
        re, im = apply_rz(re, im, post_rz2[q], q, nq)

    if double_post_flag:
        for q in range(nq):
            re, im = apply_ry(re, im, post2_ry[q], q, nq)
            re, im = apply_rz(re, im, post2_rz[q], q, nq)

    return _state_to_probs_real(re, im)
