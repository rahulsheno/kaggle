import numpy as np, torch, time, sys
import torch.nn as nn
import _utils as u
from _utils import warp_batch, hblur_batch, corrupt_batch, make_features_batch, nrmse

X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
X_DIG = np.load("X_dig_MNIST.npz")["arr_0"].astype(np.float64) / 255.0
Y_DIG = np.load("y_dig_MNIST.npz")["arr_0"].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG]); YALL = np.concatenate([Y_TRAIN, Y_DIG])
_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64); CN = _a["calib_noisy"].astype(np.float64); CC = _a["calib_clean"].astype(np.float64)

MIX = (0.80, 25.0, 110.0, 0.0, 200.0)
torch.set_num_threads(8)

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

class DenNet(nn.Module):
    def __init__(self, nch, w=56, nblk=7):
        super().__init__()
        layers = [nn.Conv2d(nch, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        return x[:, :1] + self.body(x)

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

DEN_A = DenNet(2); DEN_A.load_state_dict(torch.load('_den_a.pt')); DEN_A.eval()
DEN_C = DenNet(4); DEN_C.load_state_dict(torch.load('_den_c.pt')); DEN_C.eval()
for m in (DEN_A, DEN_C):
    for p in m.parameters():
        p.requires_grad = False

def denoise_ensemble(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = DEN_A(torch.from_numpy(FA)).numpy()[:, 0]
        RC = DEN_C(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig

rng = np.random.default_rng(17)
torch.manual_seed(17)

def gen_mixed(N):
    n_c = int(N * 0.35); n_d = N - n_c
    idx = rng.integers(0, len(XALL), N)
    X0 = np.clip(warp_batch(XALL[idx], rng.uniform(-15, 15, N), rng.uniform(0.85, 1.15, N),
                            rng.uniform(-2.5, 2.5, N), rng.uniform(-2.5, 2.5, N)), 0, 1)
    Ls = rng.choice(np.array([1, 1, 1, 3]), n_c)
    Xc = hblur_batch(X0[:n_c], Ls)
    nlev = rng.uniform(0, 0.08, n_c)
    Xc = np.clip(Xc + rng.uniform(-0.08, 0.08, (n_c, 1, 1))
                 + rng.normal(0, 1, (n_c, 28, 28)) * nlev[:, None, None], 0, 1)
    Zz, _, SIGD = corrupt_batch(X0[n_c:], rng, t_mix=MIX)
    Xd, _ = denoise_ensemble(Zz)
    # random extra noise on some denoised samples to teach robustness
    extra = rng.uniform(0, 0.05, (n_d, 1, 1))
    Xd = np.clip(Xd + rng.normal(0, 1, (n_d, 28, 28)) * extra, 0, 1)
    X = np.concatenate([Xc, Xd])
    S = np.concatenate([nlev, np.clip(SIGD + extra[:, 0, 0], 0, 1.2)])
    perm = rng.permutation(N)
    inp = np.empty((N, 2, 28, 28), dtype=np.float32)
    inp[:, 0] = X[perm]; inp[:, 1] = S[perm][:, None, None]
    return torch.from_numpy(inp), YALL[idx][perm]

def gen_noisy4(N):
    idx = rng.integers(0, len(XALL), N)
    Zz, _, _ = corrupt_batch(XALL[idx], rng, t_mix=MIX)
    F, _ = make_features_C_batch(Zz)
    return torch.from_numpy(F), YALL[idx]

crit = nn.CrossEntropyLoss(label_smoothing=0.05)

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

# ---------- anchor fine-tune ----------
ANCH_IDX = np.array([8 * r + j for r in range(64) for j in (1, 2)])
ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10 for r in range(64) for j in (1, 2)])

def augment_crops(Zc, rngf, reps):
    outs = []
    for _ in range(reps):
        th = rngf.uniform(-8, 8, len(Zc))
        sc = rngf.uniform(0.92, 1.08, len(Zc))
        dx = rngf.uniform(-1.5, 1.5, len(Zc)); dy = rngf.uniform(-1.5, 1.5, len(Zc))
        W = warp_batch(Zc, th, sc, dx, dy)
        W = W + rngf.normal(0, 1, W.shape) * rngf.uniform(0, 0.12, (len(Zc), 1, 1))
        W = W + rngf.uniform(-0.05, 0.05, (len(Zc), 1, 1))
        outs.append(np.clip(W, -0.35, 1.35))
    return np.concatenate(outs)

def finetune_anchors(net, view, reps=48, epochs=3, lr=1e-4, batch=128, name='ft'):
    rngf = np.random.default_rng(99)
    ZA = Z[ANCH_IDX]
    X = augment_crops(ZA, rngf, reps)
    Y = np.tile(ANCH_LAB, reps)
    perm = rngf.permutation(len(X))
    X, Y = X[perm], Y[perm]
    if view == 'mixed':
        Rd, _ = denoise_ensemble(X)
        _, SIGX = make_features_batch(X)
        inp = np.empty((len(X), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rd; inp[:, 1] = SIGX[:, None, None]
    else:
        inp, _ = make_features_C_batch(X)
    INP = torch.from_numpy(inp)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    crit0 = nn.CrossEntropyLoss()
    nper = len(X) // batch
    for ep in range(epochs):
        tot = 0; cor = 0
        for k in range(nper):
            sl = slice(k * batch, (k + 1) * batch)
            out = net(INP[sl])
            loss = crit0(out, torch.from_numpy(Y[sl]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * batch; cor += int((out.argmax(1).numpy() == Y[sl]).sum())
        print(f"{name} anchor-ft epoch {ep}: acc {cor / (nper * batch):.4f}", flush=True)
    return net

def anchor_acc(net, view):
    ZA = Z[ANCH_IDX]
    if view == 'mixed':
        Rd, _ = denoise_ensemble(ZA)
        _, SIGA = make_features_batch(ZA)
        inp = np.empty((len(ZA), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rd; inp[:, 1] = SIGA[:, None, None]
    else:
        inp, _ = make_features_C_batch(ZA)
    with torch.no_grad():
        o = net(torch.from_numpy(inp)).numpy()
    return (o.argmax(1) == ANCH_LAB).mean()

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('mixed', 'all'):
        net_c = train(Cls(2), gen_mixed, 30000, 8, 128, 'mixed')
        torch.save(net_c.state_dict(), '_cls_mixed_v3.pt')
        print('mixed pre-ft anchor acc:', anchor_acc(net_c, 'mixed'), flush=True)
        net_c = finetune_anchors(net_c, 'mixed', name='mixed')
        torch.save(net_c.state_dict(), '_cls_mixed_v3ft.pt')
        print('mixed post-ft anchor acc:', anchor_acc(net_c, 'mixed'), flush=True)
    if which in ('noisy', 'all'):
        net_n = train(Cls(4), gen_noisy4, 30000, 8, 128, 'noisy4')
        torch.save(net_n.state_dict(), '_cls_noisy_v3.pt')
        print('noisy pre-ft anchor acc:', anchor_acc(net_n, 'noisy'), flush=True)
        net_n = finetune_anchors(net_n, 'noisy', name='noisy4')
        torch.save(net_n.state_dict(), '_cls_noisy_v3ft.pt')
        print('noisy post-ft anchor acc:', anchor_acc(net_n, 'noisy'), flush=True)
    print('classifier step done', flush=True)
