import time
import numpy as np
import torch
import torch.nn as nn
import _utils as u
from _utils import warp_batch, hblur_batch, corrupt_batch, make_features_batch

torch.set_num_threads(8)

X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
Y_TRAIN = np.load('y_kannada_MNIST_train.npz')['arr_0'].astype(np.int64)
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
Y_DIG = np.load('y_dig_MNIST.npz')['arr_0'].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG]); YALL = np.concatenate([Y_TRAIN, Y_DIG])
_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
CN = _a['calib_noisy'].astype(np.float64)


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


from _dev15_cls_final import DenNet

DEN_A = DenNet(2); DEN_A.load_state_dict(torch.load('_den_a.pt')); DEN_A.eval()
DEN_C = DenNet(4); DEN_C.load_state_dict(torch.load('_den_c.pt')); DEN_C.eval()
for m in (DEN_A, DEN_C):
    for p in m.parameters():
        p.requires_grad = False


def restore(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = DEN_A(torch.from_numpy(FA)).numpy()[:, 0]
        RC = DEN_C(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig


class ResBlk(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.a = nn.Conv2d(w, w, 3, padding=1)
        self.b1 = nn.BatchNorm2d(w)
        self.c = nn.Conv2d(w, w, 3, padding=1)
        self.b2 = nn.BatchNorm2d(w)

    def forward(self, x):
        y = self.b1(torch.relu(self.a(x)))
        y = self.b2(self.c(y))
        return torch.relu(x + y)


class Cls2(nn.Module):
    def __init__(self, nch):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(nch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(2))
        self.lift = nn.Conv2d(64, 128, 1)
        self.s2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                ResBlk(128), nn.MaxPool2d(2))
        self.s3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                ResBlk(256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        self.head = nn.Linear(256, 10)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(self.lift(x))
        x = self.s3(x)
        x = self.pool(x).flatten(1)
        return self.head(self.drop(x))


ANCH_IDX = np.array([8 * r + j for r in range(64) for j in (1, 2)])
ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10
                     for r in range(64) for j in (1, 2)])

_rng = np.random.default_rng(31)
torch.manual_seed(31)
_CRIT = nn.CrossEntropyLoss(label_smoothing=0.1)


def gen_mixed(N):
    n_c = int(N * 0.30); n_d = N - n_c
    idx = _rng.integers(0, len(XALL), N)
    X0 = np.clip(warp_batch(XALL[idx], _rng.uniform(-15, 15, N), _rng.uniform(0.85, 1.15, N),
                            _rng.uniform(-2.5, 2.5, N), _rng.uniform(-2.5, 2.5, N)), 0, 1)
    Ls = _rng.choice(np.array([1, 1, 1, 3]), n_c)
    Xc = hblur_batch(X0[:n_c], Ls)
    nlev = _rng.uniform(0, 0.08, n_c)
    Xc = np.clip(Xc + _rng.uniform(-0.08, 0.08, (n_c, 1, 1))
                 + _rng.normal(0, 1, (n_c, 28, 28)) * nlev[:, None, None], 0, 1)
    Zz, _, SIGD = corrupt_batch(X0[n_c:], _rng)
    Xd, _ = restore(Zz)
    W = warp_batch(Xd, _rng.uniform(-6, 6, n_d), _rng.uniform(0.94, 1.06, n_d),
                   _rng.uniform(-1.5, 1.5, n_d), _rng.uniform(-1.5, 1.5, n_d))
    extra = _rng.uniform(0, 0.05, (n_d, 1, 1))
    Xd = np.clip(W + _rng.uniform(-0.06, 0.06, (n_d, 1, 1))
                 + _rng.normal(0, 1, (n_d, 28, 28)) * extra, 0, 1)
    X = np.concatenate([Xc, Xd])
    S = np.concatenate([nlev, np.clip(SIGD + extra[:, 0, 0], 0, 1.2)])
    perm = _rng.permutation(N)
    inp = np.empty((N, 2, 28, 28), dtype=np.float32)
    inp[:, 0] = X[perm]; inp[:, 1] = S[perm][:, None, None]
    return torch.from_numpy(inp), YALL[idx][perm]


def gen_noisy4(N):
    idx = _rng.integers(0, len(XALL), N)
    Zz, _, _ = corrupt_batch(XALL[idx], _rng)
    F, _ = make_features_C_batch(Zz)
    return torch.from_numpy(F), YALL[idx]


def anchor_eval(net, mk):
    inp, _ = mk(Z[ANCH_IDX])
    with torch.no_grad():
        o = net(torch.from_numpy(inp)).numpy()
    return float((o.argmax(1) == ANCH_LAB).mean())


def mk_mixed(Zz):
    Rd, sig = restore(Zz)
    inp = np.empty((len(Zz), 2, 28, 28), dtype=np.float32)
    inp[:, 0] = Rd; inp[:, 1] = sig[:, None, None]
    return inp, sig


def train(net, gen, N, epochs, batch, name, aev=None, val=None):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-5)
    best = (-1.0, None)
    for ep in range(epochs):
        t0 = time.time()
        INP, Y = gen(N)
        perm = _rng.permutation(N)
        tot = 0; cor = 0
        for k in range(nper):
            sl = perm[k * batch:(k + 1) * batch]
            out = net(INP[sl])
            loss = _CRIT(out, torch.from_numpy(Y[sl]))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch; cor += int((out.argmax(1).numpy() == Y[sl]).sum())
        msg = f'[{name}] ep{ep}: loss {tot / N:.4f} acc {cor / N:.4f} ({time.time() - t0:.0f}s)'
        if aev is not None:
            a = anchor_eval(net, aev)
            sa = 0.0
            if val is not None:
                VIN, VY = val
                with torch.no_grad():
                    vo = net(VIN).argmax(1).numpy()
                sa = float((vo == VY).mean())
            msg += f' anchor {a:.4f} sim {sa:.4f}'
            score = 0.5 * a + 0.5 * sa
            if score > best[0]:
                best = (score, {k_: t.detach().clone() for k_, t in net.state_dict().items()})
        print(msg, flush=True)
    if aev is not None and best[1] is not None:
        net.load_state_dict(best[1])
        print(f'[{name}] kept best combined {best[0]:.4f}', flush=True)
    return net


def make_sim_val(n=1600, seed=777):
    rngv = np.random.default_rng(seed)
    idx = rngv.integers(0, len(XALL), n)
    X0 = np.clip(warp_batch(XALL[idx], rngv.uniform(-15, 15, n), rngv.uniform(0.85, 1.15, n),
                            rngv.uniform(-2.5, 2.5, n), rngv.uniform(-2.5, 2.5, n)), 0, 1)
    Zz, _, _ = corrupt_batch(X0, rngv)
    return Zz, YALL[idx]


VAL_Z, VAL_Y = make_sim_val()
VAL_MIXED = None
VAL_NOISY = None


def lazy_vals():
    global VAL_MIXED, VAL_NOISY
    if VAL_MIXED is None:
        inp, _ = mk_mixed(VAL_Z)
        VAL_MIXED = (torch.from_numpy(inp), VAL_Y)
        FIN, _ = make_features_C_batch(VAL_Z)
        VAL_NOISY = (torch.from_numpy(FIN), VAL_Y)


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    lazy_vals()
    if which in ('mixed', 'all'):
        net = train(Cls2(2), gen_mixed, 24000, 7, 128, 'mixed2',
                    aev=mk_mixed, val=VAL_MIXED)
        torch.save(net.state_dict(), '_cls_mixed2.pt')
    if which in ('noisy', 'all'):
        net = train(Cls2(4), gen_noisy4, 24000, 7, 128, 'noisy2',
                    aev=lambda Zz: make_features_C_batch(Zz), val=VAL_NOISY)
        torch.save(net.state_dict(), '_cls_noisy2.pt')
    print('cls v2 done', flush=True)
