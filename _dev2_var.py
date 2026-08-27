import numpy as np, torch, time, pickle, sys
import torch.nn as nn
exec(open('_dev1_den.py').read().split('if __name__')[0])

BM, BT = pickle.load(open('_sigcorr.pkl', 'rb'))
def sig_correct(s):
    return float(np.interp(s, BM, BT, left=BM[0] * 0, right=BT[-1]))

def desalt2(z):
    med = median_filter(z, size=3, mode='nearest')
    out = z.copy()
    bad = ((z <= -0.349) & (med > -0.10)) | ((z >= 1.349) & (med < 0.60))
    out[bad] = med[bad]
    return out

def make_features(z, mode):
    sig = sig_correct(sigma_mad(z))
    zd = desalt2(z)
    clipmask = (z <= -0.349) | (z >= 1.349)
    zv = zd[~clipmask]
    if len(zv) >= 200:
        mu = np.percentile(zv, 15)
        hi = np.percentile(zv, 97)
        sc = max(hi - mu, 3 * sig, 0.25)
    else:
        mu, sc = 0.0, 1.0
    zn = np.clip((zd - mu) / sc, -1.5, 2.5)
    sigc = np.full((28, 28), sig)
    if mode == 'A':
        return np.stack([zd, sigc]).astype(np.float32), sig
    if mode == 'B':
        return np.stack([zn, sigc / sc]).astype(np.float32), sig
    if mode == 'C':
        return np.stack([zn, zd, sigc, clipmask.astype(np.float64)]).astype(np.float32), sig

class Net(nn.Module):
    def __init__(self, nch, w=48, nblk=6):
        super().__init__()
        layers = [nn.Conv2d(nch, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)
    def forward(self, x):
        return x[:, :1] + self.body(x)

def run_variant(mode, N=10000, epochs=2, batch=64, w=48):
    nch = {'A': 2, 'B': 2, 'C': 4}[mode]
    net = Net(nch, w=w)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
    crit = nn.MSELoss()
    # calibration inputs fixed
    CIN = torch.from_numpy(np.stack([make_features(z, mode)[0] for z in CN]))
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.integers(0, len(XALL), N)
        Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
        INP = np.stack([make_features(z, mode)[0] for z in Zz])
        tot = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            out = net(torch.from_numpy(INP[sl]))
            loss = crit(out, torch.from_numpy(X0[sl].astype(np.float32))[:, None])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch
        with torch.no_grad():
            yh = net(CIN).numpy().clip(0, 1)
        print(f"[{mode}] epoch {ep}: mse {tot / N:.5f} calib-NRMSE {nrmse(yh[:, 0], CC):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'C'
    run_variant(mode)
