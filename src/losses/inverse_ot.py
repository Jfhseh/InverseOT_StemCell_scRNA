"""
Bilevel Inverse Optimal Transport loss for hyperbolic crypt representation learning.

Formulation (Shi et al. ICML 2023, adapted for crypt-axis setting):

    Inner:  pi^theta = argmin_{pi in U(a,b)} <C^theta, pi> - eps * H(pi)
            C^theta_ij = d_H(z_i, z_j)^2   (squared hyperbolic distance)
            Solved via log-domain Sinkhorn with uniform marginals.

    Outer:  L_IOT = KL(pi_target || pi^theta) + regularization

Three target coupling modes (set via coupling_mode):

  "gaussian"   (within-batch, smooth)
      pi_target[i,j] ∝ exp(-|depth_i - depth_j|^2 / sigma_depth^2)
      Soft match on continuous depth values.  Blurs zone boundaries.

  "zone_block" (prototype / cross-donor)
      Snap each cell to its discrete MROI zone bin {0,1,2,3}.
      pi_target[i,j] ∝ zone_compat[zone(i), zone(j)]
      where zone_compat = exp(-|zone_i - zone_j|^2 / sigma_zone^2).

      Creates block structure: ALL cells in the same zone bin — regardless
      of donor, age, or region — receive identical coupling weight.  This
      is the "prototype = zone" abstraction.  Cross-donor alignment emerges
      naturally: a sub-crypt stem cell from a 0-week mouse and one from a
      2-year mouse both land in zone 0, so the model is forced to embed
      them in the same hyperbolic neighbourhood.

  "radial"     (zone_block target + radius-only cost)
      Cost: C[i,j] = (||z_i|| - ||z_j||)²  (squared difference in radius).
      Target: zone_block coupling (same zone → high mass).

      Effect: same-zone cells are pushed to the same radius; different-zone
      cells are pushed to different radii.  No angular pressure — the IOT
      loss cannot cause angular collapse or origin collapse.

      This mode generalises across donors because the radius is a scalar
      signal: once the model learns zone-k → radius band r_k on training
      donors, held-out donors' cells with the same zone labels land at the
      same radius bands.  Use with lambda_crypt > 0 to anchor the absolute
      radius value for each zone.

Regularization to prevent dimension collapse (from the robotics paper):
    L_unif:  Wang-Isola uniformity — log-sum-exp of -t*||v_i - v_j||^2
             in the tangent space; minimizing this spreads embeddings out.
    L_var:   penalizes per-dimension std < gamma (prevents dimension collapse).
    L_cov:   penalizes off-diagonal covariance (encourages decorrelated dims).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .iot_targets import build_multiscale_target_coupling, depth_to_zone


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn inner loop
# ---------------------------------------------------------------------------

def _log_sinkhorn(
    log_C: torch.Tensor,
    n_iter: int = 30,
) -> torch.Tensor:
    """
    Log-domain Sinkhorn with uniform marginals a = b = 1/B.

    Parameters
    ----------
    log_C  : (B, B) tensor — log of the kernel matrix (= -C / epsilon).
    n_iter : number of Sinkhorn iterations.

    Returns
    -------
    log_pi : (B, B) log-coupling tensor.
    """
    B = log_C.shape[0]
    log_r = -torch.full((B,), float(B), device=log_C.device,
                        dtype=log_C.dtype).log()   # log(1/B)

    log_u = torch.zeros(B, device=log_C.device, dtype=log_C.dtype)
    log_v = torch.zeros(B, device=log_C.device, dtype=log_C.dtype)

    for _ in range(n_iter):
        # log u ← log(1/B) - LSE_j( log_C + log_v )
        log_u = log_r - torch.logsumexp(log_C + log_v.unsqueeze(0), dim=1)
        # log v ← log(1/B) - LSE_i( log_C + log_u )
        log_v = log_r - torch.logsumexp(log_C + log_u.unsqueeze(1), dim=0)

    return log_u.unsqueeze(1) + log_C + log_v.unsqueeze(0)   # (B, B)


# ---------------------------------------------------------------------------
# Target coupling construction — two modes
# ---------------------------------------------------------------------------

# SCP2595 MROI zone boundaries (depth values are 0, 0.25, 0.5, 1.0)
_ZONE_BOUNDARIES = [0.125, 0.375, 0.75]   # cut-points for zones 0→1, 1→2, 2→3


def _depth_to_zone(depths: torch.Tensor) -> torch.Tensor:
    """Map continuous depths in [0,1] to integer zone indices {0,1,2,3}."""
    zones = torch.zeros(len(depths), dtype=torch.long, device=depths.device)
    for i, boundary in enumerate(_ZONE_BOUNDARIES):
        zones[depths >= boundary] = i + 1
    return zones


def _build_gaussian_coupling(
    depths: torch.Tensor,
    sigma: float,
    n_iter: int = 50,
) -> torch.Tensor:
    """
    Smooth target coupling: pi_target[i,j] ∝ exp(-|depth_i - depth_j|^2 / sigma^2).
    Made doubly stochastic via Sinkhorn (no grad).
    """
    with torch.no_grad():
        diff_sq = (depths.unsqueeze(1) - depths.unsqueeze(0)) ** 2
        log_raw = -diff_sq / (sigma ** 2)
        return _log_sinkhorn(log_raw, n_iter=n_iter).exp()


def _build_zone_block_coupling(
    depths: torch.Tensor,
    sigma_zone: float = 0.8,
    n_iter: int = 50,
) -> torch.Tensor:
    """
    Prototype-level target coupling based on discrete MROI zone bins.

    Each cell is assigned to one of 4 zone bins:
        0 = sub-crypt  (depth < 0.125)
        1 = crypt base (0.125 ≤ depth < 0.375)
        2 = crypt mid  (0.375 ≤ depth < 0.75)
        3 = crypt apex (depth ≥ 0.75)

    pi_target[i,j] ∝ exp(-|zone(i) - zone(j)|^2 / sigma_zone^2)

    Block structure: ALL cells in the same zone get identical coupling,
    independent of donor, age, or region.  Alignment across donors is
    implicit — a stem cell from a 0w mouse and one from a 2yr mouse both
    land in zone 0 and are forced to neighbour each other in hyperbolic space.

    With sigma_zone=0.8:
        same zone  → 1.00
        +1 zone    → 0.21   (adjacent)
        +2 zones   → 0.002  (skip-one)
        +3 zones   → ~0     (opposite ends)
    """
    with torch.no_grad():
        zones = _depth_to_zone(depths).float()                        # (B,)
        zone_diff = (zones.unsqueeze(1) - zones.unsqueeze(0)).abs()   # (B, B)
        log_raw = -zone_diff.pow(2) / (sigma_zone ** 2)
        return _log_sinkhorn(log_raw, n_iter=n_iter).exp()


# ---------------------------------------------------------------------------
# Regularization terms
# ---------------------------------------------------------------------------

def _uniformity_loss(v: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Wang-Isola uniformity: log-mean of Gaussian kernel over pairwise distances.
    Minimizing this spreads points out in the tangent space.
    """
    sq = torch.cdist(v, v).pow(2)   # (B, B)
    return torch.logsumexp(-t * sq.reshape(-1), dim=0) - sq.numel().bit_length()


def _variance_loss(v: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Penalize any dimension whose std < gamma."""
    sigma = v.std(dim=0)            # (D,)
    return F.relu(gamma - sigma).pow(2).mean()


def _covariance_loss(v: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal entries of the empirical covariance matrix."""
    B, D = v.shape
    v_c = v - v.mean(dim=0, keepdim=True)
    cov = (v_c.T @ v_c) / (B - 1)  # (D, D)
    off = ~torch.eye(D, dtype=torch.bool, device=v.device)
    return cov[off].pow(2).mean()


def _radial_variance_loss(z: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Penalize low variance in the hyperbolic radius ||z||."""
    radii = z.norm(dim=-1)
    sigma = radii.std()
    return F.relu(gamma - sigma).pow(2)


# ---------------------------------------------------------------------------
# Prototype-level distribution alignment  (K×K, the principled version)
# ---------------------------------------------------------------------------

def _zone_prototype_ot_loss(
    z: torch.Tensor,
    zones: torch.Tensor,
    manifold,
    epsilon: float = 0.1,
    n_sink_iter: int = 20,
    sigma_zone: float = 0.8,
    sub_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Distribution-level IOT: align the K=4 zone distributions (or K' zone x sub_label).
    """
    device = z.device
    dtype  = z.dtype

    if sub_labels is not None:
        # Group by zone (0-3) and sub_label (e.g. age 0-10)
        # We assume sub_labels is an integer tensor (e.g. 0 to N). We multiply zone by 100 to avoid collision.
        group_ids = zones * 100 + sub_labels
    else:
        group_ids = zones

    unique_groups = torch.unique(group_ids)

    protos = []
    proto_zones = []
    for g in unique_groups:
        mask = (group_ids == g)
        if mask.sum() < 1:
            continue
        protos.append(z[mask].mean(dim=0))   # (D,) — grad flows through mean
        # retrieve original zone for target coupling
        proto_zones.append((g // 100).item() if sub_labels is not None else g.item())

    n_protos = len(protos)
    if n_protos < 2:
        return z.sum() * 0.0, {"iot_kl": 0.0}

    z_proto = torch.stack(protos, dim=0)                             # (n_protos, D)

    # K×K cost matrix of squared hyperbolic distances between prototypes
    C_proto = manifold.dist(
        z_proto.unsqueeze(1), z_proto.unsqueeze(0)
    ).pow(2)                                                          # (n_protos, n_protos)

    with torch.no_grad():
        C_med = C_proto.median().clamp(min=1e-4)
    log_K_mat = -C_proto / (C_med * epsilon)

    log_pi_theta = _log_sinkhorn(log_K_mat, n_iter=n_sink_iter)      # (n_protos, n_protos)

    # Target coupling: zone compatibility between present prototypes
    keys_t = torch.tensor(proto_zones, dtype=dtype, device=device)
    zone_diff = (keys_t.unsqueeze(1) - keys_t.unsqueeze(0)).abs()
    log_pi_target_raw = -zone_diff.pow(2) / (sigma_zone ** 2)
    with torch.no_grad():
        log_pi_target = _log_sinkhorn(log_pi_target_raw, n_iter=50)
        pi_target = log_pi_target.exp()

    l_kl = -(pi_target * log_pi_theta).sum()
    return l_kl, {"iot_kl": l_kl.detach().item()}


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

def inverse_ot_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    manifold,
    epsilon: float = 0.1,
    n_sink_iter: int = 20,
    coupling_mode: str = "gaussian",
    sigma_depth: float = 0.15,
    sigma_zone: float = 0.8,
    lambda_unif: float = 0.1,
    lambda_var: float = 0.25,
    lambda_cov: float = 0.1,
    gamma_var: float = 1.0,
    lambda_rad_var: float = 0.0,
    gamma_rad_var: float = 0.1,
    prototype_grouping: str = "zone",
    sub_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Bilevel inverse OT loss for crypt-axis representation learning.

    Parameters
    ----------
    z             : (B, D) hyperbolic embeddings in the Poincaré ball.
    labels        : (B,) crypt-axis depth values in [0, 1].  NaN entries skipped.
    manifold      : geoopt.PoincareBall instance.
    epsilon       : entropic regularization for Sinkhorn (relative to median cost).
    n_sink_iter   : inner Sinkhorn iterations.
    coupling_mode : "gaussian"      — smooth depth-similarity, cell-level B×B.
                    "zone_block"    — discrete zone block, cell-level B×B.
                    "zone_proto"    — zone prototype distribution alignment, K×K.
                                      This is the principled cross-distribution mode:
                                      OT operates on the K=4 zone distributions, not
                                      individual cells.  Cross-donor alignment is
                                      implicit via prototype mean aggregation.
                    "radial"        — zone_block target + radius-only cost.
                                      C[i,j] = (||z_i|| - ||z_j||)².  No angular
                                      pressure; use with lambda_crypt>0 to anchor depths.
    sigma_depth   : Gaussian coupling bandwidth (coupling_mode="gaussian").
    sigma_zone    : Zone coupling bandwidth in zone units ("zone_block"/"zone_proto").
                    0.8 → same-zone=1.0, adjacent=0.21, skip-one=0.002.
    lambda_unif   : weight for Wang-Isola uniformity term.
    lambda_var    : weight for per-dimension variance penalty.
    lambda_cov    : weight for off-diagonal covariance penalty.
    gamma_var     : target minimum std per dimension.

    Returns
    -------
    (total_loss, component_dict)
    """
    # Filter NaN labels
    valid = ~torch.isnan(labels)
    if valid.sum() < 4:
        dummy = z.sum() * 0.0
        return dummy, {"iot_kl": 0.0, "iot_unif": 0.0, "iot_var": 0.0, "iot_cov": 0.0, "iot_rad_var": 0.0}

    z_v = z[valid]
    y_v = labels[valid]
    sub_v = sub_labels[valid] if sub_labels is not None else None

    # --- Zone prototype mode: K×K distribution alignment ---
    if coupling_mode == "zone_proto":
        zones = _depth_to_zone(y_v)
        # Use sub_v if grouping is zone_age, else None
        active_sub_labels = sub_v if prototype_grouping == "zone_age" else None
        l_kl, kl_comps = _zone_prototype_ot_loss(
            z_v, zones, manifold,
            epsilon=epsilon, n_sink_iter=n_sink_iter, sigma_zone=sigma_zone,
            sub_labels=active_sub_labels
        )
        # Regularisation still on all cells
        v = manifold.logmap0(z_v)
        l_unif = _uniformity_loss(v)
        l_var  = _variance_loss(v, gamma=gamma_var)
        l_cov  = _covariance_loss(v)
        
        l_rad_var = torch.zeros(1, device=z.device)
        if lambda_rad_var > 0:
            l_rad_var = _radial_variance_loss(z_v, gamma=gamma_rad_var)

        total = l_kl + lambda_unif * l_unif + lambda_var * l_var + lambda_cov * l_cov + lambda_rad_var * l_rad_var
        return total, {
            "iot_kl":   kl_comps["iot_kl"],
            "iot_unif": l_unif.detach().item(),
            "iot_var":  l_var.detach().item(),
            "iot_cov":  l_cov.detach().item(),
            "iot_rad_var": l_rad_var.detach().item() if lambda_rad_var > 0 else 0.0,
        }

    # --- Cell-level modes: B×B ---
    B = z_v.shape[0]

    # 1. Cost matrix and target coupling
    if coupling_mode == "radial":
        # Cost: squared difference in hyperbolic radius (Euclidean norm of Poincaré point).
        # No angular component — IOT cannot collapse points to origin or cause
        # angular zone encoding.  Pair with lambda_crypt>0 to anchor absolute depths.
        radii = z_v.norm(dim=-1)                                          # (B,)
        C = (radii.unsqueeze(1) - radii.unsqueeze(0)).pow(2)              # (B, B)
        pi_target = _build_zone_block_coupling(y_v, sigma_zone=sigma_zone)
    elif coupling_mode == "gaussian":
        hyp_dist = manifold.dist(z_v.unsqueeze(1), z_v.unsqueeze(0))     # (B, B)
        C = hyp_dist.pow(2)
        pi_target = _build_gaussian_coupling(y_v, sigma=sigma_depth)
    elif coupling_mode == "zone_block":
        hyp_dist = manifold.dist(z_v.unsqueeze(1), z_v.unsqueeze(0))     # (B, B)
        C = hyp_dist.pow(2)
        pi_target = _build_zone_block_coupling(y_v, sigma_zone=sigma_zone)
    else:
        raise ValueError(f"Unknown coupling_mode '{coupling_mode}'. "
                         f"Choose 'gaussian', 'zone_block', 'zone_proto', or 'radial'.")

    with torch.no_grad():
        C_med = C.median().clamp(min=1e-4)
    C_norm = C / C_med

    # 2. Inner loop: entropic OT
    log_K = -C_norm / epsilon
    log_pi_theta = _log_sinkhorn(log_K, n_iter=n_sink_iter)

    # 3. Outer KL
    l_kl = -(pi_target * log_pi_theta).sum()

    # 5. Regularisation
    v = manifold.logmap0(z_v)
    l_unif = _uniformity_loss(v)
    l_var  = _variance_loss(v, gamma=gamma_var)
    l_cov  = _covariance_loss(v)

    l_rad_var = torch.zeros(1, device=z.device)
    if lambda_rad_var > 0:
        l_rad_var = _radial_variance_loss(z_v, gamma=gamma_rad_var)

    total = l_kl + lambda_unif * l_unif + lambda_var * l_var + lambda_cov * l_cov + lambda_rad_var * l_rad_var
    return total, {
        "iot_kl":   l_kl.detach().item(),
        "iot_unif": l_unif.detach().item(),
        "iot_var":  l_var.detach().item(),
        "iot_cov":  l_cov.detach().item(),
        "iot_rad_var": l_rad_var.detach().item() if lambda_rad_var > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Phase 2B: Multi-scale inverse OT loss
# ---------------------------------------------------------------------------

def multiscale_inverse_ot_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    expr_batch: torch.Tensor,
    manifold,
    epsilon: float = 0.1,
    n_sink_iter: int = 20,
    tau_zone: float = 1.0,
    tau_expr: float = 1.0,
    w_zone: float = 1.0,
    w_adjacent: float = 0.5,
    w_expr: float = 0.25,
    max_zone_gap: int = 1,
    n_star_sink_iter: int = 50,
    donor_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Phase 2B bilevel inverse OT with multi-scale target coupling.

    Inner:  pi^theta = Sinkhorn(C^theta / eps)
            C^theta[i,j] = d_H(z_i, z_j)^2

    Outer:  L = -sum_ij P_star[i,j] * log pi^theta[i,j]

    P_star is a multi-scale coupling (see iot_targets.py):
        P_star = normalize(w_zone * P_zone + w_adj * P_adjacent + w_expr * P_expr)

    Unlike the V1 zone_block mode, P_expr provides within-zone transcript
    structure signal so the IOT objective does not fully sacrifice NP@15.

    The unif/var/cov regularizers from V1 are NOT applied here; they are
    replaced by the explicit radial geometry losses in radial_geometry.py.

    Parameters
    ----------
    z           : (B, D) hyperbolic embeddings in Poincaré ball.
    labels      : (B,) crypt depth in [0,1]. NaN entries skipped.
    expr_batch  : (B, D_expr) expression features (used for P_expr component).
    manifold    : geoopt.PoincareBall instance.
    epsilon     : Sinkhorn entropic regularization (relative to median cost).
    n_sink_iter : inner Sinkhorn iterations.
    tau_zone    : zone adjacency bandwidth for P_adjacent.
    tau_expr    : expression bandwidth for P_expr (relative to batch median).
    w_zone      : weight for P_zone component.
    w_adjacent  : weight for P_adjacent component.
    w_expr      : weight for P_expr component.
    max_zone_gap: max zone distance for P_expr mask.
    n_star_sink_iter: Sinkhorn iterations to normalize P_star.

    Returns
    -------
    (total_loss, {"iot_kl": float})
    """
    valid = ~torch.isnan(labels)
    if valid.sum() < 4:
        return z.sum() * 0.0, {"iot_kl": 0.0}

    z_v    = z[valid]
    y_v    = labels[valid]
    expr_v = expr_batch[valid]

    zones_v = depth_to_zone(y_v)
    donors_v = donor_ids[valid] if donor_ids is not None else None

    # Build multi-scale target coupling (no grad)
    pi_target = build_multiscale_target_coupling(
        zones_v, expr_v,
        donor_ids=donors_v,
        tau_zone=tau_zone, tau_expr=tau_expr,
        w_zone=w_zone, w_adjacent=w_adjacent, w_expr=w_expr,
        max_zone_gap=max_zone_gap,
        n_sink_iter=n_star_sink_iter,
    )                                                                  # (B_v, B_v)

    # Hyperbolic cost matrix (with grad)
    C = manifold.dist(z_v.unsqueeze(1), z_v.unsqueeze(0)).pow(2)     # (B_v, B_v)

    with torch.no_grad():
        C_med = C.median().clamp(min=1e-4)
    log_K = -C / (C_med * epsilon)

    log_pi_theta = _log_sinkhorn(log_K, n_iter=n_sink_iter)          # (B_v, B_v)

    # Normalize by B so IOT is per-cell (same scale as crypt MSE / triplet mean).
    # Without this, the B×B sum is ~B times larger than per-cell losses, causing
    # IOT to dominate regardless of lambda weighting.
    l_kl = -(pi_target * log_pi_theta).sum() / len(z_v)

    return l_kl, {"iot_kl": l_kl.detach().item()}
