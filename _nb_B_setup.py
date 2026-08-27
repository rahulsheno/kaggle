# ==================== SETUP - forward model & noise estimation ====================
# Vectorized versions of the GIVEN chain (bit-identical per image), plus a per-crop
# noise-level estimator calibrated by running the chain over the official corpus.
import time
import numpy as np
from scipy.ndimage import median_filter

SEED = 7
MIX = (0.80, 25.0, 110.0, 0.0, 200.0)   # t sampling: 80% U(25,110) + 20% U(0,200)


def warp_batch(S, theta_deg, scale, dx, dy):
    """Batched, bit-identical to the given warp() per image."""
    S = np.asarray(S)
    B, h, w = S.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    th = np.deg2rad(theta_deg)
    ct = np.cos(th) / scale
    st = np.sin(th) / scale
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ry = yy[None] - cy - dy[:, None, None]
    rx = xx[None] - cx - dx[:, None, None]
    sx = ct[:, None, None] * rx + st[:, None, None] * ry + cx
    sy = -st[:, None, None] * rx + ct[:, None, None] * ry + cy
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx = sx - x0; fy = sy - y0
    out = np.zeros_like(S, dtype=np.float64)
    bi = np.arange(B)[:, None, None]
    for oy in (0, 1):
        for ox in (0, 1):
            yi, xi = y0 + oy, x0 + ox
            ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            wgt = (fy if oy else 1 - fy) * (fx if ox else 1 - fx)
            vals = S[bi, np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
            out += np.where(ok, wgt * vals, 0.0)
    return out


def hblur_batch(imgs, Ls):
    out = np.empty_like(imgs)
    for L in np.unique(Ls):
        L = int(L)
        mk = Ls == L
        if L <= 1:
            out[mk] = imgs[mk]
            continue
        p = L // 2
        pad = np.pad(imgs[mk], ((0, 0), (0, 0), (p, p)), mode="edge")
        acc = 0
        for k in range(L):
            acc = acc + pad[:, :, k:k + imgs.shape[2]]
        out[mk] = acc / L
    return out


def corrupt_batch(S, rng, t_mix=MIX):
    """The given chain, vectorized: s -> warp -> optics -> diffusion+read noise -> bit rot -> clip."""
    S = np.asarray(S)
    B = len(S)
    th = rng.uniform(-15, 15, B)
    sc = rng.uniform(0.85, 1.15, B)
    dx = rng.uniform(-2.5, 2.5, B)
    dy = rng.uniform(-2.5, 2.5, B)
    Ls = rng.choice(np.array([1, 3, 5, 7, 9]), B, p=[0.42, 0.30, 0.18, 0.07, 0.03])
    g = rng.uniform(0.45, 1.15, B)
    b = rng.uniform(-0.02, 0.25, B)
    frac, a, b2, c, d2 = t_mix
    m = rng.random(B)
    t = np.empty(B)
    t[m < frac] = rng.uniform(a, b2, int((m < frac).sum()))
    t[m >= frac] = rng.uniform(c, d2, int((m >= frac).sum()))
    psp = np.where(rng.random(B) < 0.75, 0.0, rng.uniform(0.01, 0.08, B))
    X0 = np.clip(warp_batch(S, th, sc, dx, dy), 0, 1)
    U = g[:, None, None] * hblur_batch(X0, Ls) + b[:, None, None]
    ab = np.array([cosine_alpha_bar(v) for v in t])
    sig = np.sqrt(1 - ab)
    Zz = np.sqrt(ab)[:, None, None] * U + sig[:, None, None] * rng.standard_normal((B, 28, 28))
    r = rng.random((B, 28, 28))
    sp = psp[:, None, None]
    Zz = np.where(sp > 0, np.where(r < sp / 2, 1.35, np.where(r < sp, -0.35, Zz)), Zz)
    return np.clip(Zz, -0.35, 1.35), X0, sig


# ---- per-crop noise level: robust MAD on vertical diffs, corrected for clip bias ----
def sigma_mad_raw_batch(Zz):
    Zz = np.asarray(Zz, dtype=np.float64)
    B = len(Zz)
    med = median_filter(Zz, size=(1, 3, 3), mode='nearest')
    bad = ((Zz <= -0.349) & (med > -0.10)) | ((Zz >= 1.349) & (med < 0.60))
    ZD = np.where(bad, med, Zz)                      # de-salt isolated bit-rot extremes
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    dv = ZD[:, 1:, :] - ZD[:, :-1, :]
    okd = ~clip0[:, 1:, :] & ~clip0[:, :-1, :]
    est = np.empty(B)
    for i in range(B):
        d = dv[i][okd[i]]
        if len(d) < 200:
            d = dv[i].ravel()
        md = np.median(d)
        est[i] = np.median(np.abs(d - md))
    return np.clip((est / 0.6745 / np.sqrt(2)) / 0.962, 0.03, 1.2), ZD


def build_sigma_correction(S, rng, n_per_t=60):
    """Empirically map the raw MAD estimate to true sigma (clipping truncates tails)."""
    ts = np.arange(5, 200, 5)
    N = n_per_t * len(ts)
    idx = rng.integers(0, len(S), N)
    Zz, X0, SIG = corrupt_batch(S[idx], rng)
    est, _ = sigma_mad_raw_batch(Zz)
    order = np.argsort(est)
    e_s, t_s = est[order], SIG[order]
    bins = np.linspace(0.05, 1.15, 45)
    bm, bt = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (e_s >= lo) & (e_s < hi)
        if m.sum() > 10:
            bm.append(e_s[m].mean()); bt.append(np.median(t_s[m]))
    return np.array(bm), np.array(bt)


_rng_setup = np.random.default_rng(SEED)
XALL = np.concatenate([X_TRAIN, X_DIG])
YALL = np.concatenate([Y_TRAIN, Y_DIG])
BM, BT = build_sigma_correction(XALL, _rng_setup)
print("sigma correction table: %d bins; est 0.30 -> %.3f, est 0.45 -> %.3f"
      % (len(BM), np.interp(0.30, BM, BT), np.interp(0.45, BM, BT)))


def sigma_estimate_batch(Zz):
    est, ZD = sigma_mad_raw_batch(Zz)
    return np.interp(est, BM, BT, left=0.0, right=BT[-1]), ZD


def make_features_A(Zz):
    """view A: [de-salted z, sigma]"""
    sig, ZD = sigma_estimate_batch(Zz)
    feat = np.empty((len(Zz), 2, 28, 28), dtype=np.float32)
    feat[:, 0] = ZD
    feat[:, 1] = sig[:, None, None]
    return feat, sig


def make_features_C(Zz):
    """view C: [contrast-normalized z, de-salted z, sigma, clipmask]"""
    sig, ZD = sigma_estimate_batch(Zz)
    Zz = np.asarray(Zz, dtype=np.float64)
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    feat = np.empty((len(Zz), 4, 28, 28), dtype=np.float32)
    for i in range(len(Zz)):
        v = ZD[i][~clip0[i]]
        if len(v) >= 200:
            mu = np.percentile(v, 15); hi = np.percentile(v, 97)
            sc = max(hi - mu, 3 * sig[i], 0.25)
        else:
            mu, sc = 0.0, 1.0
        feat[i, 0] = np.clip((ZD[i] - mu) / sc, -1.5, 2.5)
        feat[i, 1] = ZD[i]
        feat[i, 2] = sig[i]
        feat[i, 3] = clip0[i].astype(np.float32)
    return feat, sig


def nrmse(yh, y):
    yh = np.asarray(yh); y = np.asarray(y)
    return float(np.sqrt(np.mean((yh - y) ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-8))


sig_archive, _ = sigma_estimate_batch(Z)
sig_calib, _ = sigma_estimate_batch(CALIB_NOISY)
print("archive sigma: med %.3f  p90 %.3f  max %.3f"
      % (np.median(sig_archive), np.percentile(sig_archive, 90), sig_archive.max()))
print("calib   sigma: med %.3f  min %.3f  max %.3f"
      % (np.median(sig_calib), sig_calib.min(), sig_calib.max()))
print("setup done")
