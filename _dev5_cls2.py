import numpy as np, torch, time
import torch.nn as nn
import torch.nn.functional as F
exec(open('_dev1_den.py').read().split('if __name__')[0])
import pickle
BM, BT = pickle.load(open('_sigcorr.pkl', 'rb'))
def sig_correct(s): return float(np.interp(s, BM, BT, left=0.0, right=BT[-1]))
def desalt2(z):
    med = median_filter(z, size=3, mode='nearest')
    out = z.copy()
    bad = ((z <= -0.349) & (med > -0.10)) | ((z >= 1.349) & (med < 0.60))
    out[bad] = med[bad]
    return out
def make_features(z):
    sig = sig_correct(sigma_mad(z))
    zd = desalt2(z)
    return np.stack([zd.astype(np.float32), np.full((28, 28), np.float32(sig))]), sig

WEIGHTS = np.array([7, 3, 1, 7, 3, 1, 7])
WARDS = [1, 2, 3, 4]
FORMULARY = {
    "05": (10, 40), "08": (5, 25), "11": (20, 60), "17": (15, 45),
    "23": (30, 90), "26": (10, 50), "34": (25, 75), "39": (5, 35),
    "42": (40, 95), "51": (20, 70), "63": (15, 55), "77": (35, 85),
}
INV = {1: 1, 3: 7, 7: 3, 9: 9}

def check_digit(body):
    return int(np.dot(WEIGHTS, np.asarray(body)) % 10)

def row_is_legal(r, row8):
    d = list(map(int, row8))
    if d[0] not in WARDS: return False
    if d[1] * 10 + d[2] != r + 1: return False
    code = f"{d[3]}{d[4]}"
    if code not in FORMULARY: return False
    lo, hi = FORMULARY[code]
    return lo <= d[5] * 10 + d[6] <= hi

class Cls(nn.Module):
    def __init__(self, nch=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(nch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(128, 10)
    def forward(self, x):
        return self.head(self.f(x))

def warp_labels(S, Y, rng, N):
    idx = rng.integers(0, len(S), N)
    X0 = np.empty((N, 28, 28))
    for i, j in enumerate(idx):
        X0[i] = np.clip(warp(S[j], rng.uniform(-15, 15), rng.uniform(0.85, 1.15),
                             rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)), 0, 1)
    return X0, Y[idx]

def train_cls_clean(N=24000, epochs=4, batch=128):
    net = Cls(1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        t0 = time.time()
        X0, Y = warp_labels(XALL, YALL, rng, N)
        Ls = rng.choice([1, 1, 1, 3], N)
        for i in range(N):
            if Ls[i] > 1: X0[i] = hblur(X0[i], int(Ls[i]))
        X0 = np.clip(X0 + rng.uniform(-0.08, 0.08, (N, 1, 1))
                     + rng.normal(0, 1, (N, 28, 28)) * rng.uniform(0, 0.08, (N, 1, 1)), 0, 1)
        tot = 0; cor = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            out = net(torch.from_numpy(X0[sl].astype(np.float32))[:, None])
            yt = torch.from_numpy(Y[sl])
            loss = crit(out, yt)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch; cor += int((out.argmax(1) == yt).sum())
        print(f"clean-cls epoch {ep}: loss {tot / N:.4f} acc {cor / N:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net

def train_cls_noisy(N=24000, epochs=4, batch=128):
    net = Cls(2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.integers(0, len(XALL), N)
        Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
        INP = np.stack([make_features(z)[0] for z in Zz])
        Y = YALL[idx]
        tot = 0; cor = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            out = net(torch.from_numpy(INP[sl]))
            yt = torch.from_numpy(Y[sl])
            loss = crit(out, yt)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch; cor += int((out.argmax(1) == yt).sum())
        print(f"noisy-cls epoch {ep}: loss {tot / N:.4f} acc {cor / N:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net

if __name__ == '__main__':
    torch.save(train_cls_clean().state_dict(), '_cls_clean.pt')
    torch.save(train_cls_noisy().state_dict(), '_cls_noisy.pt')
    print('classifiers saved')
