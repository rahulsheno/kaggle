import numpy as np, torch, time, sys
import torch.nn as nn
import torch.nn.functional as F
exec(open('_dev1_den.py').read().split('if __name__')[0])

WEIGHTS = np.array([7, 3, 1, 7, 3, 1, 7])
WARDS = [1, 2, 3, 4]
FORMULARY = {
    "05": (10, 40), "08": (5, 25), "11": (20, 60), "17": (15, 45),
    "23": (30, 90), "26": (10, 50), "34": (25, 75), "39": (5, 35),
    "42": (40, 95), "51": (20, 70), "63": (15, 55), "77": (35, 85),
}

def warp_batched(S, rng):
    B = len(S)
    th = rng.uniform(-15, 15, B); sc = rng.uniform(0.85, 1.15, B)
    dx = rng.uniform(-2.5, 2.5, B); dy = rng.uniform(-2.5, 2.5, B)
    X0 = np.empty_like(S)
    for i in range(B):
        X0[i] = np.clip(warp(S[i], th[i], sc[i], dx[i], dy[i]), 0, 1)
    return X0

class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(128, 10)
    def forward(self, x):
        return self.head(self.f(x))

def train_classifier(N=24000, epochs=4, batch=128):
    net = Cls()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.integers(0, len(XALL), N)
        X0 = warp_batched(XALL[idx], rng)
        Y = YALL[idx]
        # mild post-restoration artefacts
        Ls = rng.choice([1, 1, 1, 3], N)
        for i in range(N):
            if Ls[i] > 1:
                X0[i] = hblur(X0[i], int(Ls[i]))
        X0 = X0 + rng.uniform(-0.08, 0.08, (N, 1, 1)) + rng.normal(0, 1, (N, 28, 28)) * rng.uniform(0, 0.08, (N, 1, 1))
        X0 = np.clip(X0, 0, 1)
        tot = 0; cor = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            xt = torch.from_numpy(X0[sl].astype(np.float32))[:, None]
            yt = torch.from_numpy(Y[sl])
            out = net(xt)
            loss = crit(out, yt)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch
            cor += int((out.argmax(1) == yt).sum())
        print(f"cls epoch {ep}: loss {tot / N:.4f} acc {cor / N:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net

def restore_all(net_den, Z):
    from scipy.ndimage import median_filter
    BM, BT = __import__('pickle').load(open('_sigcorr.pkl', 'rb'))
    def sig_correct(s): return float(np.interp(s, BM, BT, left=0.0, right=BT[-1]))
    def desalt2(z):
        med = median_filter(z, size=3, mode='nearest')
        out = z.copy()
        bad = ((z <= -0.349) & (med > -0.10)) | ((z >= 1.349) & (med < 0.60))
        out[bad] = med[bad]
        return out
    outs = np.empty((len(Z), 28, 28))
    sigs = np.empty(len(Z))
    with torch.no_grad():
        for i, z in enumerate(Z):
            sig = sig_correct(sigma_mad(z))
            zd = desalt2(z)
            inp = np.stack([zd.astype(np.float32), np.full((28, 28), np.float32(sig))])
            outs[i] = net_den(torch.from_numpy(inp)[None])[0, 0].numpy().clip(0, 1)
            sigs[i] = sig
    return outs, sigs

if __name__ == '__main__':
    net_cls = train_classifier()
    torch.save(net_cls.state_dict(), '_cls.pt')
    print('classifier saved')
