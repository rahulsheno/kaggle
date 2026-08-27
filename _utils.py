import numpy as np
import pickle
from scipy.ndimage import median_filter

T_STEPS = 200

def cosine_alpha_bar(t, T=T_STEPS, s=0.008):
    f = lambda u: np.cos(((u / T) + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(f(np.asarray(t, dtype=np.float64)) / f(0), 1e-6, 1.0)

def warp_batch(S, theta_deg, scale, dx, dy):
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

def warp(img, theta_deg, scale, dx, dy):
    return warp_batch(img[None], np.array([theta_deg]), np.array([scale]),
                      np.array([dx]), np.array([dy]))[0]

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

def hblur(img, L):
    return hblur_batch(img[None], np.array([L]))[0]

def corrupt_batch(S, rng, t_mix=(0.85, 15, 90, 0, 200)):
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
    ab = cosine_alpha_bar(t)
    sig = np.sqrt(1 - ab)
    Zz = np.sqrt(ab)[:, None, None] * U + sig[:, None, None] * rng.standard_normal((B, 28, 28))
    r = rng.random((B, 28, 28))
    sp = psp[:, None, None]
    Zz = np.where(sp > 0, np.where(r < sp / 2, 1.35, np.where(r < sp, -0.35, Zz)), Zz)
    Zz = np.clip(Zz, -0.35, 1.35)
    return Zz, X0, sig

def sigma_correction():
    BM, BT = pickle.load(open('_sigcorr.pkl', 'rb'))
    def f(est):
        return np.interp(np.asarray(est, dtype=np.float64), BM, BT, left=0.0, right=BT[-1])
    return f

SIG_CORRECT = sigma_correction()

def make_features_batch(Zz):
    Zz = np.asarray(Zz, dtype=np.float64)
    B = len(Zz)
    med = median_filter(Zz, size=(1, 3, 3), mode='nearest')
    bad = ((Zz <= -0.349) & (med > -0.10)) | ((Zz >= 1.349) & (med < 0.60))
    ZD = np.where(bad, med, Zz)
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
    est = np.clip((est / 0.6745 / np.sqrt(2)) / 0.962, 0.03, 1.2)
    sig = SIG_CORRECT(est)
    feat = np.empty((B, 2, 28, 28), dtype=np.float32)
    feat[:, 0] = ZD
    feat[:, 1] = sig[:, None, None]
    return feat, sig

def make_features(z):
    f, s = make_features_batch(np.asarray(z)[None])
    return f[0], float(s[0])

def nrmse(yh, y):
    yh = np.asarray(yh); y = np.asarray(y)
    return float(np.sqrt(np.mean((yh - y) ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-8))
