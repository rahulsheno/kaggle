# ==================== TRACK B - two readers ====================
# reader 1 (mixed): restored image + sigma channel; training mix of lightly degraded
#                   corpus images and archive-chain corruptions passed through the
#                   denoiser ensemble (with mild post-restoration warps), so it learns
#                   exactly what restoration leaves behind.
# reader 2 (noisy4): raw archive view (contrast-normalized z, de-salted z, sigma, clipmask);
#                   never trusts the denoiser at all.
# Both are residual BatchNorm CNNs. Epochs are selected on a blend of bed-anchor accuracy
# (the only labelled handwriting of the actual scribe) and held-out simulated accuracy.
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


class Cls(nn.Module):
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
        return self.head(self.drop(self.pool(x).flatten(1)))


ANCH_IDX = np.array([8 * r + j for r in range(N_ROWS) for j in (1, 2)])
ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10
                     for r in range(N_ROWS) for j in (1, 2)])

_rng_cls = np.random.default_rng(31)
torch.manual_seed(31)
_CRIT = nn.CrossEntropyLoss(label_smoothing=0.1)
CLS_N_TRAIN = 24000


def gen_mixed(N):
    n_c = int(N * 0.30); n_d = N - n_c
    idx = _rng_cls.integers(0, len(XALL), N)
    X0 = np.clip(warp_batch(XALL[idx], _rng_cls.uniform(-15, 15, N),
                            _rng_cls.uniform(0.85, 1.15, N),
                            _rng_cls.uniform(-2.5, 2.5, N),
                            _rng_cls.uniform(-2.5, 2.5, N)), 0, 1)
    Ls = _rng_cls.choice(np.array([1, 1, 1, 3]), n_c)
    Xc = hblur_batch(X0[:n_c], Ls)
    nlev = _rng_cls.uniform(0, 0.08, n_c)
    Xc = np.clip(Xc + _rng_cls.uniform(-0.08, 0.08, (n_c, 1, 1))
                 + _rng_cls.normal(0, 1, (n_c, 28, 28)) * nlev[:, None, None], 0, 1)
    Zz, _, SIGD = corrupt_batch(X0[n_c:], _rng_cls)
    Xd, _ = restore_batch(Zz)
    W = warp_batch(Xd, _rng_cls.uniform(-6, 6, n_d), _rng_cls.uniform(0.94, 1.06, n_d),
                   _rng_cls.uniform(-1.5, 1.5, n_d), _rng_cls.uniform(-1.5, 1.5, n_d))
    extra = _rng_cls.uniform(0, 0.05, (n_d, 1, 1))
    Xd = np.clip(W + _rng_cls.uniform(-0.06, 0.06, (n_d, 1, 1))
                 + _rng_cls.normal(0, 1, (n_d, 28, 28)) * extra, 0, 1)
    X = np.concatenate([Xc, Xd])
    S = np.concatenate([nlev, np.clip(SIGD + extra[:, 0, 0], 0, 1.2)])
    perm = _rng_cls.permutation(N)
    inp = np.empty((N, 2, 28, 28), dtype=np.float32)
    inp[:, 0] = X[perm]; inp[:, 1] = S[perm][:, None, None]
    return torch.from_numpy(inp), YALL[idx][perm]


def gen_noisy4(N):
    idx = _rng_cls.integers(0, len(XALL), N)
    Zz, _, _ = corrupt_batch(XALL[idx], _rng_cls)
    F, _ = make_features_C(Zz)
    return torch.from_numpy(F), YALL[idx]


def _view_mixed(Zz):
    Rd, sig = restore_batch(Zz)
    inp = np.empty((len(Zz), 2, 28, 28), dtype=np.float32)
    inp[:, 0] = Rd; inp[:, 1] = sig[:, None, None]
    return inp


def anchor_eval(net, view):
    Zi = Z[ANCH_IDX]
    inp = _view_mixed(Zi) if view == 'mixed' else make_features_C(Zi)[0]
    with torch.no_grad():
        o = net(torch.from_numpy(inp)).numpy()
    return float((o.argmax(1) == ANCH_LAB).mean())


def make_sim_val(n=1600, seed=777):
    rngv = np.random.default_rng(seed)
    idx = rngv.integers(0, len(XALL), n)
    X0 = np.clip(warp_batch(XALL[idx], rngv.uniform(-15, 15, n), rngv.uniform(0.85, 1.15, n),
                            rngv.uniform(-2.5, 2.5, n), rngv.uniform(-2.5, 2.5, n)), 0, 1)
    Zz, _, _ = corrupt_batch(X0, rngv)
    return Zz, YALL[idx]


VAL_Z, VAL_Y = make_sim_val()
VAL_M = torch.from_numpy(_view_mixed(VAL_Z))
VAL_N = torch.from_numpy(make_features_C(VAL_Z)[0])


def train_cls(net, gen, epochs, batch, name, view, val_inp):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = CLS_N_TRAIN // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-5)
    best = (-1.0, None)
    for ep in range(epochs):
        t0 = time.time()
        INP, Y = gen(CLS_N_TRAIN)
        perm = _rng_cls.permutation(CLS_N_TRAIN)
        tot = 0; cor = 0
        for k in range(nper):
            sl = perm[k * batch:(k + 1) * batch]
            out = net(INP[sl])
            loss = _CRIT(out, torch.from_numpy(Y[sl]))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += loss.item() * batch; cor += int((out.argmax(1).numpy() == Y[sl]).sum())
        a = anchor_eval(net, view)
        with torch.no_grad():
            sa = float((net(val_inp).argmax(1).numpy() == VAL_Y).mean())
        print(f"  [{name}] epoch {ep}: loss {tot / CLS_N_TRAIN:.4f} acc {cor / CLS_N_TRAIN:.4f} "
              f"anchor {a:.4f} sim {sa:.4f} ({time.time() - t0:.0f}s)", flush=True)
        score = 0.5 * a + 0.5 * sa
        if score > best[0]:
            best = (score, {k: t.detach().clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print(f"  [{name}] kept best combined {best[0]:.4f}", flush=True)
    return net


CLS_M = train_cls(Cls(2), gen_mixed, 7, 128, 'mixed', 'mixed', VAL_M)
CLS_N = train_cls(Cls(4), gen_noisy4, 7, 128, 'noisy4', 'noisy', VAL_N)
print("anchor acc: mixed %.4f | noisy4 %.4f" % (anchor_eval(CLS_M, 'mixed'),
                                                 anchor_eval(CLS_N, 'noisy')))
