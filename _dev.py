import os, sys, time
import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(7)
torch.manual_seed(7)
torch.set_num_threads(8)

# ---------------- data ----------------
def load():
    X_train = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
    y_train = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
    X_dig = np.load("X_dig_MNIST.npz")["arr_0"].astype(np.float64) / 255.0
    y_dig = np.load("y_dig_MNIST.npz")["arr_0"].astype(np.int64)
    a = np.load("arogya_archive_v1.npz")
    return dict(X=X_train, Y=y_train, XD=X_dig, YD=y_dig,
                Z=a["Z"].astype(np.float64),
                CN=a["calib_noisy"].astype(np.float64),
                CC=a["calib_clean"].astype(np.float64))

D = load()
Z, CN, CC = D["Z"], D["CN"], D["CC"]
XALL = np.concatenate([D["X"], D["XD"]])
YALL = np.concatenate([D["Y"], D["YD"]])

T_STEPS = 200
def cosine_alpha_bar(t, T=T_STEPS, s=0.008):
    f = lambda u: np.cos(((u / T) + s) / (1 + s) * np.pi / 2) ** 2
    return float(np.clip(f(t) / f(0), 1e-6, 1.0))
TS = np.arange(T_STEPS + 1)
AB = np.array([cosine_alpha_bar(t) for t in TS])

def warp(img, theta_deg, scale, dx, dy):
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th) / scale, np.sin(th) / scale
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ry, rx = yy - cy - dy, xx - cx - dx
    sx = ct * rx + st * ry + cx
    sy = -st * rx + ct * ry + cy
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    out = np.zeros_like(img, dtype=np.float64)
    for oy in (0, 1):
        for ox in (0, 1):
            yi, xi = y0 + oy, x0 + ox
            ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            wgt = (fy if oy else 1 - fy) * (fx if ox else 1 - fx)
            out[ok] += wgt[ok] * img[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)][ok]
    return out

def hblur(img, L):
    if L <= 1:
        return img.copy()
    p = L // 2
    pad = np.pad(img, ((0, 0), (p, p)), mode="edge")
    return sum(pad[:, k:k + img.shape[1]] for k in range(L)) / L

LCHOICE = [1, 3, 5, 7, 9]
LPROB = [0.42, 0.30, 0.18, 0.07, 0.03]

def corrupt_batch(S, rng):
    """S: (B,28,28) clean sources -> returns z, x0, sig (noise std), params"""
    B = len(S)
    th = rng.uniform(-15, 15, B)
    sc = rng.uniform(0.85, 1.15, B)
    dx = rng.uniform(-2.5, 2.5, B)
    dy = rng.uniform(-2.5, 2.5, B)
    Ls = rng.choice(LCHOICE, B, p=LPROB)
    g = rng.uniform(0.45, 1.15, B)
    b = rng.uniform(-0.02, 0.25, B)
    # noise schedule: concentrate where the archive lives, keep tails
    t = np.empty(B)
    m = rng.random(B)
    t[m < 0.85] = rng.uniform(15, 90, (m < 0.85).sum())
    t[m >= 0.85] = rng.uniform(0, 200, (m >= 0.85).sum())
    psp = np.where(rng.random(B) < 0.75, 0.0, rng.uniform(0.01, 0.08, B))
    X0 = np.empty_like(S); Zz = np.empty_like(S)
    for i in range(B):
        x0 = np.clip(warp(S[i], th[i], sc[i], dx[i], dy[i]), 0, 1)
        u = g[i] * hblur(x0, int(Ls[i])) + b[i]
        ab = cosine_alpha_bar(t[i])
        sig = np.sqrt(1 - ab)
        z = np.sqrt(ab) * u + sig * rng.standard_normal((28, 28))
        if psp[i] > 0:
            r = rng.random((28, 28))
            z = np.where(r < psp[i] / 2, 1.35, np.where(r < psp[i], -0.35, z))
        X0[i] = x0
        Zz[i] = np.clip(z, -0.35, 1.35)
    return Zz, X0, np.sqrt(1 - np.array([cosine_alpha_bar(v) for v in t]))

# ---------------- normalization & sigma estimation ----------------
def desalt(z, sigma):
    med = ndi_median(z)
    out = z.copy()
    thr = max(0.9, 6 * sigma)
    atom_lo = z <= -0.349
    atom_hi = z >= 1.349
    bad = (atom_lo & (med > -0.349 + 0.4)) | (atom_hi & (med < 1.349 - 0.4))
    bad &= np.abs(z - med) > thr * 0 + 0.55
    out[bad] = med[bad]
    return out

from scipy.ndimage import median_filter
def ndi_median(z):
    return median_filter(z, size=3, mode="nearest")

def sigma_mad(z):
    clipmask = (z <= -0.349) | (z >= 1.349)
    dv = z[1:, :] - z[:-1, :]
    ok = ~clipmask[1:, :] & ~clipmask[:-1, :]
    d = dv[ok] if ok.sum() > 200 else dv.ravel()
    mad = np.median(np.abs(d - np.median(d)))
    return float(np.clip((mad / 0.6745 / np.sqrt(2)) / 0.962, 0.03, 1.2))

def normalize(z, sig):
    clipmask = (z <= -0.349) | (z >= 1.349)
    zv = z[~clipmask]
    if len(zv) < 200:
        mu, sc = 0.0, 1.0
    else:
        mu = np.percentile(zv, 15)
        hi = np.percentile(zv, 97)
        sc = max(hi - mu, 3 * sig, 0.25)
    zn = (z - mu) / sc
    return np.clip(zn, -1.5, 2.5), sig / sc

def make_input(z):
    sig = sigma_mad(z)
    zd = desalt(z, sig)
    zn, sign = normalize(zd, sig)
    inp = np.stack([zn, np.full((28, 28), sign)], 0)
    return inp.astype(np.float32), sig

# ---------------- model ----------------
class Denoiser(nn.Module):
    def __init__(self, w=48, nblk=6):
        super().__init__()
        layers = [nn.Conv2d(2, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        zch = x[:, :1]
        r = self.body(x)
        return torch.clamp(zch * 0 + torch.sigmoid(zch + r) * 0 + (zch + r) * 0.0 + zch, -2, 3)  # placeholder

print("module check")
