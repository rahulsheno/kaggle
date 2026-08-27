import numpy as np, torch, time
import torch.nn as nn
import _utils as u
from _utils import warp_batch, hblur_batch, corrupt_batch, make_features_batch

X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
X_DIG = np.load("X_dig_MNIST.npz")["arr_0"].astype(np.float64) / 255.0
Y_DIG = np.load("y_dig_MNIST.npz")["arr_0"].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG]); YALL = np.concatenate([Y_TRAIN, Y_DIG])

rng = np.random.default_rng(7)
torch.manual_seed(7)
torch.set_num_threads(8)

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

DEN = DenNet(2, 56, 7)
DEN.load_state_dict(torch.load('_denoiser_final.pt'))
DEN.eval()
for p in DEN.parameters():
    p.requires_grad = False

def denoise(Zz):
    F, _ = make_features_batch(Zz)
    with torch.no_grad():
        R = DEN(torch.from_numpy(F)).numpy()[:, 0]
    return np.clip(R, 0, 1)

def gen_mixed(N):
    n_c = int(N * 0.4); n_d = N - n_c
    idx = rng.integers(0, len(XALL), N)
    X0 = np.clip(warp_batch(XALL[idx], rng.uniform(-15, 15, N), rng.uniform(0.85, 1.15, N),
                            rng.uniform(-2.5, 2.5, N), rng.uniform(-2.5, 2.5, N)), 0, 1)
    # clean branch: mild artefacts
    Ls = rng.choice(np.array([1, 1, 1, 3]), n_c)
    Xc = hblur_batch(X0[:n_c], Ls)
    nlev = rng.uniform(0, 0.08, (n_c, 1, 1))
    Xc = np.clip(Xc + rng.uniform(-0.08, 0.08, (n_c, 1, 1))
                 + rng.normal(0, 1, (n_c, 28, 28)) * nlev, 0, 1)
    SIGC = nlev[:, 0, 0]
    # denoised branch: full chain -> denoiser
    Zz, _, SIGD = corrupt_batch(X0[n_c:], rng)
    Xd = denoise(Zz)
    X = np.concatenate([Xc, Xd])
    S = np.concatenate([SIGC, SIGD])
    perm = rng.permutation(N)
    inp = np.empty((N, 2, 28, 28), dtype=np.float32)
    inp[:, 0] = X[perm]
    inp[:, 1] = S[perm][:, None, None]
    return torch.from_numpy(inp), YALL[idx][perm]

def gen_noisy_feats(N):
    idx = rng.integers(0, len(XALL), N)
    Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
    F, sig = make_features_batch(Zz)
    # add normalized channel: clipmask + zn
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    ZD = F[:, 0].astype(np.float64)
    feat = np.empty((N, 4, 28, 28), dtype=np.float32)
    for i in range(N):
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
    return torch.from_numpy(feat), YALL[idx]

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

if __name__ == '__main__':
    net_c = train(Cls(2), gen_mixed, 30000, 6, 128, 'mixed')
    torch.save(net_c.state_dict(), '_cls_mixed.pt')
    net_n = train(Cls(4), gen_noisy_feats, 30000, 6, 128, 'noisy4')
    torch.save(net_n.state_dict(), '_cls_noisy4.pt')
    print('saved')
