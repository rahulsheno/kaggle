import numpy as np, torch, time, sys
import torch.nn as nn
import _utils as u
from _utils import corrupt_batch, make_features_batch, nrmse

X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
X_DIG = np.load("X_dig_MNIST.npz")["arr_0"].astype(np.float64) / 255.0
Y_DIG = np.load("y_dig_MNIST.npz")["arr_0"].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG])
_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64); CN = _a["calib_noisy"].astype(np.float64); CC = _a["calib_clean"].astype(np.float64)

torch.set_num_threads(8)

def make_features_C_batch(Zz):
    F_A, sig = make_features_batch(Zz)          # [zd, sig]
    B = len(Zz)
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    ZD = F_A[:, 0].astype(np.float64)
    feat = np.empty((B, 4, 28, 28), dtype=np.float32)
    for i in range(B):
        v = ZD[i][~clip0[i]]
        if len(v) >= 200:
            mu = np.percentile(v, 15)
            hi = np.percentile(v, 97)
            sc = max(hi - mu, 3 * sig[i], 0.25)
        else:
            mu, sc = 0.0, 1.0
        feat[i, 0] = np.clip((ZD[i] - mu) / sc, -1.5, 2.5)
        feat[i, 1] = ZD[i]
        feat[i, 2] = sig[i]
        feat[i, 3] = clip0[i].astype(np.float32)
    return feat, sig

class Net(nn.Module):
    def __init__(self, nch, w=56, nblk=7):
        super().__init__()
        layers = [nn.Conv2d(nch, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)
    def forward(self, x):
        return x[:, :1] + self.body(x)

def train_model(name, seed, variant, N=24000, epochs=4, batch=64):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nch = 2 if variant == 'A' else 4
    mk = make_features_batch if variant == 'A' else make_features_C_batch
    net = Net(nch)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
    crit = nn.MSELoss()
    CIN = torch.from_numpy(mk(CN)[0])
    best = (1e9, None)
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.integers(0, len(XALL), N)
        Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
        INP = mk(Zz)[0]
        tot = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            out = net(torch.from_numpy(INP[sl]))
            loss = crit(out, torch.from_numpy(X0[sl].astype(np.float32))[:, None])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch
        with torch.no_grad():
            yh = net(CIN).numpy()[:, 0].clip(0, 1)
        v = nrmse(yh, CC)
        print(f"[{name}] epoch {ep}: mse {tot / N:.5f} calib-NRMSE {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1])
    torch.save(best[1], f"_den_{name}.pt")
    print(f"[{name}] saved best calib-NRMSE {best[0]:.4f}", flush=True)
    return net

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('a', 'both'):
        train_model('a', 7, 'A')
    if which in ('c', 'both'):
        train_model('c', 11, 'C')
