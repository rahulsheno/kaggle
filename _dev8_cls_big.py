import numpy as np, torch, time
import torch.nn as nn
import _utils as u
from _utils import warp_batch, hblur_batch, corrupt_batch, make_features_batch

X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
X_DIG = np.load("X_dig_MNIST.npz")["arr_0"].astype(np.float64) / 255.0
Y_DIG = np.load("y_dig_MNIST.npz")["arr_0"].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG]); YALL = np.concatenate([Y_TRAIN, Y_DIG])
_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64); CN = _a["calib_noisy"].astype(np.float64); CC = _a["calib_clean"].astype(np.float64)

rng = np.random.default_rng(7)
torch.manual_seed(7)
torch.set_num_threads(8)

class Cls(nn.Module):
    def __init__(self, nch=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(nch, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(192, 10)
    def forward(self, x):
        return self.head(self.f(x))

crit = nn.CrossEntropyLoss(label_smoothing=0.05)

def gen_clean(N):
    idx = rng.integers(0, len(XALL), N)
    X0 = np.clip(warp_batch(XALL[idx], rng.uniform(-15, 15, N), rng.uniform(0.85, 1.15, N),
                            rng.uniform(-2.5, 2.5, N), rng.uniform(-2.5, 2.5, N)), 0, 1)
    Ls = rng.choice(np.array([1, 1, 1, 3]), N)
    X0 = hblur_batch(X0, Ls)
    X0 = np.clip(X0 + rng.uniform(-0.08, 0.08, (N, 1, 1))
                 + rng.normal(0, 1, (N, 28, 28)) * rng.uniform(0, 0.08, (N, 1, 1)), 0, 1)
    return torch.from_numpy(X0.astype(np.float32))[:, None], YALL[idx]

def gen_noisy(N):
    idx = rng.integers(0, len(XALL), N)
    Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
    F, _ = make_features_batch(Zz)
    return torch.from_numpy(F), YALL[idx]

def train(net, gen, N, epochs, batch, name):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
    for ep in range(epochs):
        t0 = time.time()
        INP, Y = gen(N)
        tot = 0; cor = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            out = net(INP[sl])
            loss = crit(out, torch.from_numpy(Y[sl]))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch; cor += int((out.argmax(1).numpy() == Y[sl]).sum())
        print(f"{name} epoch {ep}: loss {tot / N:.4f} acc {cor / N:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return net

net_c = train(Cls(1), gen_clean, 30000, 8, 128, 'clean')
torch.save(net_c.state_dict(), '_cls_clean.pt')
net_n = train(Cls(2), gen_noisy, 30000, 6, 128, 'noisy')
torch.save(net_n.state_dict(), '_cls_noisy.pt')

net_c.eval(); net_n.eval()
with torch.no_grad():
    o = net_c(torch.from_numpy(X_TRAIN[:8000].astype(np.float32))[:, None]).numpy()
print('clean-cls raw X_TRAIN[:8000] acc:', (o.argmax(1) == Y_TRAIN[:8000]).mean())
with torch.no_grad():
    o = net_c(torch.from_numpy(X_DIG.astype(np.float32))[:, None]).numpy()
print('clean-cls raw X_DIG acc:', (o.argmax(1) == Y_DIG).mean())
Zz, X0, SIG = corrupt_batch(X_TRAIN[:3000], np.random.default_rng(5))
F, _ = make_features_batch(Zz)
with torch.no_grad():
    o = net_n(torch.from_numpy(F)).numpy()
for lo, hi in [(0, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.6), (0.6, 2)]:
    mk = (SIG >= lo) & (SIG < hi)
    if mk.sum():
        print(f"noisy-cls sigma [{lo:.2f},{hi:.2f}): acc {(o[mk].argmax(1) == Y_TRAIN[:3000][mk]).mean():.3f} n={mk.sum()}")
