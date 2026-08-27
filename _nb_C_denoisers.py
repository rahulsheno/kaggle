# ==================== TRACK A - denoiser ensemble ====================
# Two residual conv nets with different input views, trained on simulated corruptions
# of the corpus (the given chain with the archive-matched noise schedule). The epoch
# with the best calibration-strip NRMSE is kept. Each net predicts the pre-optics
# image x0 as a residual on its first input channel.
import os
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


class DenNet(nn.Module):
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


def train_denoiser(name, seed, nch, mk, N=24000, epochs=5, batch=64):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = DenNet(nch)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    nper = N // batch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-5)
    crit = nn.MSELoss()
    CIN = torch.from_numpy(mk(CALIB_NOISY)[0])
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
        v = nrmse(yh, CALIB_CLEAN)
        print(f"  [{name}] epoch {ep}: mse {tot / N:.5f} calib-NRMSE {v:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if v < best[0]:
            best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print(f"  [{name}] kept best calib-NRMSE {best[0]:.4f}", flush=True)
    return net


DEN_A = train_denoiser("den-A", 7, 2, make_features_A)
DEN_C = train_denoiser("den-C", 11, 4, make_features_C)


def restore_batch(Zz):
    FA, sig = make_features_A(Zz)
    FC, _ = make_features_C(Zz)
    with torch.no_grad():
        RA = DEN_A(torch.from_numpy(FA)).numpy()[:, 0]
        RC = DEN_C(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig


REST_CAL, _ = restore_batch(CALIB_NOISY)
print("ensemble calib NRMSE %.4f (all-zero baseline 1.000)" % nrmse(REST_CAL, CALIB_CLEAN))
