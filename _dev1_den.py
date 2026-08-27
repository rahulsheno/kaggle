import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter

SEED = 7
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(8)

# ---------------- corpus + archive ----------------
X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
X_DIG = np.load("X_dig_MNIST.npz")["arr_0"].astype(np.float64) / 255.0
Y_DIG = np.load("y_dig_MNIST.npz")["arr_0"].astype(np.int64)
_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64)
CN = _a["calib_noisy"].astype(np.float64)
CC = _a["calib_clean"].astype(np.float64)
XALL = np.concatenate([X_TRAIN, X_DIG])
YALL = np.concatenate([Y_TRAIN, Y_DIG])

T_STEPS = 200
def cosine_alpha_bar(t, T=T_STEPS, s=0.008):
    f = lambda u: np.cos(((u / T) + s) / (1 + s) * np.pi / 2) ** 2
    return float(np.clip(f(t) / f(0), 1e-6, 1.0))

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
    B = len(S)
    th = rng.uniform(-15, 15, B)
    sc = rng.uniform(0.85, 1.15, B)
    dx = rng.uniform(-2.5, 2.5, B)
    dy = rng.uniform(-2.5, 2.5, B)
    Ls = rng.choice(LCHOICE, B, p=LPROB)
    g = rng.uniform(0.45, 1.15, B)
    b = rng.uniform(-0.02, 0.25, B)
    t = np.empty(B)
    m = rng.random(B)
    t[m < 0.85] = rng.uniform(15, 90, int((m < 0.85).sum()))
    t[m >= 0.85] = rng.uniform(0, 200, int((m >= 0.85).sum()))
    psp = np.where(rng.random(B) < 0.75, 0.0, rng.uniform(0.01, 0.08, B))
    X0 = np.empty_like(S); Zz = np.empty_like(S); SIG = np.empty(B)
    for i in range(B):
        x0 = np.clip(warp(S[i], th[i], sc[i], dx[i], dy[i]), 0, 1)
        u = g[i] * hblur(x0, int(Ls[i])) + b[i]
        ab = cosine_alpha_bar(t[i])
        sig = float(np.sqrt(1 - ab))
        z = np.sqrt(ab) * u + sig * rng.standard_normal((28, 28))
        if psp[i] > 0:
            r = rng.random((28, 28))
            z = np.where(r < psp[i] / 2, 1.35, np.where(r < psp[i], -0.35, z))
        X0[i] = x0; Zz[i] = np.clip(z, -0.35, 1.35); SIG[i] = sig
    return Zz, X0, SIG

def desalt(z):
    med = median_filter(z, size=3, mode="nearest")
    out = z.copy()
    atom_lo = z <= -0.349
    atom_hi = z >= 1.349
    bad = (atom_lo & (med > 0.05)) | (atom_hi & (med < 1.0))
    out[bad] = med[bad]
    return out

def sigma_mad(z):
    clipmask = (z <= -0.349) | (z >= 1.349)
    dv = z[1:, :] - z[:-1, :]
    ok = ~clipmask[1:, :] & ~clipmask[:-1, :]
    d = dv[ok] if ok.sum() > 200 else dv.ravel()
    mad = np.median(np.abs(d - np.median(d)))
    return float(np.clip((mad / 0.6745 / np.sqrt(2)) / 0.962, 0.03, 1.2))

def to_input(z):
    sig = sigma_mad(z)
    zd = desalt(z)
    clipmask = (zd <= -0.349) | (zd >= 1.349)
    zv = zd[~clipmask]
    if len(zv) >= 200:
        mu = np.percentile(zv, 15)
        hi = np.percentile(zv, 97)
        sc = max(hi - mu, 3 * sig, 0.25)
    else:
        mu, sc = 0.0, 1.0
    zn = np.clip((zd - mu) / sc, -1.5, 2.5)
    return np.stack([zn, np.full((28, 28), sig / sc)]).astype(np.float32), sig

class Denoiser(nn.Module):
    def __init__(self, w=48, nblk=6):
        super().__init__()
        layers = [nn.Conv2d(2, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        zn = x[:, :1]
        r = self.body(x)
        return torch.clamp(zn + r, 0.0, 1.0)

def nrmse(yh, y):
    return float(np.sqrt(np.mean((yh - y) ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-8))

# ---------------- training ----------------
def train_denoiser(N=16000, epochs=3, batch=64, w=48):
    net = Denoiser(w=w)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * N // batch, 1e-4)
    crit = nn.MSELoss()
    nper = N // batch
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.integers(0, len(XALL), N)
        S = XALL[idx]
        Zz, X0, SIG = corrupt_batch(S, rng)
        INP = np.stack([to_input(z)[0] for z in Zz])
        total = 0.0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            xt = torch.from_numpy(INP[sl])
            yt = torch.from_numpy(X0[sl].astype(np.float32))[:, None]
            out = net(xt)
            loss = crit(out, yt)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            total += float(loss) * batch
        print(f"epoch {ep}: train-mse {total / (nper * batch):.5f}  ({time.time() - t0:.0f}s)", flush=True)
    return net

def restore_image(net, z):
    inp, sig = to_input(z)
    with torch.no_grad():
        out = net(torch.from_numpy(inp)[None])
    return out[0, 0].numpy()

if __name__ == "__main__":
    t0 = time.time()
    net = train_denoiser()
    torch.save(net.state_dict(), "_denoiser.pt")
    # probe on calibration
    yh = np.stack([restore_image(net, z) for z in CN])
    print("calib NRMSE after sim-train: %.4f" % nrmse(yh, CC))
    print("calib per-image:", np.round([nrmse(yh[i], CC[i]) for i in range(24)], 3))
    # raw clip baseline
    print("raw clip NRMSE: %.4f" % nrmse(np.clip(CN, 0, 1), CC))
    print("total %.0fs" % (time.time() - t0))
