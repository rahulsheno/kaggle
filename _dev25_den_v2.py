import time
import numpy as np
import torch
import torch.nn as nn
import _utils as u
from _utils import corrupt_batch, make_features_batch, nrmse

torch.set_num_threads(8)

X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
Y_TRAIN = np.load('y_kannada_MNIST_train.npz')['arr_0'].astype(np.int64)
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
Y_DIG = np.load('y_dig_MNIST.npz')['arr_0'].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG])
_a = np.load('arogya_archive_v1.npz')
CN = _a['calib_noisy'].astype(np.float64)
CC = _a['calib_clean'].astype(np.float64)


def make_features_C_batch(Zz):
    F_A, sig = make_features_batch(Zz)
    B = len(Zz)
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    ZD = F_A[:, 0].astype(np.float64)
    feat = np.empty((B, 4, 28, 28), dtype=np.float32)
    for i in range(B):
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


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.a = nn.Conv2d(w, w, 3, padding=1)
        self.b = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(x)))


class DenNet2(nn.Module):
    def __init__(self, nch, w=64, ngrp=4):
        super().__init__()
        self.inp = nn.Conv2d(nch, w, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(w) for _ in range(ngrp)])
        self.out = nn.Conv2d(w, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return x[:, :1] + self.out(self.body(torch.relu(self.inp(x))))


def train(name, seed, nch, mk, N=32000, epochs=6, batch=64, lr=8e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = DenNet2(nch)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-5)
    crit = nn.MSELoss()
    CIN = torch.from_numpy(mk(CN)[0])
    best = (1e9, None)
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.integers(0, len(XALL), N)
        Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
        INP = torch.from_numpy(mk(Zz)[0])
        TOT = torch.from_numpy(X0.astype(np.float32))[:, None]
        perm = rng.permutation(N)
        tot = 0
        for k in range(nper):
            sl = perm[k * batch:(k + 1) * batch]
            out = net(INP[sl])
            loss = crit(out, TOT[sl])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch
        with torch.no_grad():
            yh = net(CIN).numpy()[:, 0].clip(0, 1)
        v = nrmse(yh, CC)
        print(f'[{name}] ep{ep}: mse {tot / N:.5f} calib {v:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if v < best[0]:
            best = (v, {k_: t.detach().clone() for k_, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    torch.save(best[1], f'_den_{name}.pt')
    print(f'[{name}] best calib {best[0]:.4f}', flush=True)
    return net


if __name__ == '__main__':
    a = train('a2', 23, 2, make_features_batch)
    c = train('c2', 29, 4, make_features_C_batch)
    FA, sig = make_features_batch(CN)
    FC, _ = make_features_C_batch(CN)
    with torch.no_grad():
        RA = a(torch.from_numpy(FA)).numpy()[:, 0]
        RC = c(torch.from_numpy(FC)).numpy()[:, 0]
    R = np.clip((RA + RC) / 2, 0, 1)
    print('ensemble calib NRMSE %.4f' % nrmse(R, CC), flush=True)
