import numpy as np, torch, time, pickle, sys
import torch.nn as nn
exec(open('_dev1_den.py').read().split('if __name__')[0])
BM, BT = pickle.load(open('_sigcorr.pkl', 'rb'))
def sig_correct(s):
    return float(np.interp(s, BM, BT, left=0.0, right=BT[-1]))

def desalt2(z):
    med = median_filter(z, size=3, mode='nearest')
    out = z.copy()
    bad = ((z <= -0.349) & (med > -0.10)) | ((z >= 1.349) & (med < 0.60))
    out[bad] = med[bad]
    return out

def make_features(z):
    sig = sig_correct(sigma_mad(z))
    zd = desalt2(z)
    sigc = np.full((28, 28), sig, dtype=np.float32)
    return np.stack([zd.astype(np.float32), sigc]), sig

class Net(nn.Module):
    def __init__(self, w=56, nblk=7):
        super().__init__()
        layers = [nn.Conv2d(2, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)
    def forward(self, x):
        return x[:, :1] + self.body(x)

def nrmse(yh, y):
    return float(np.sqrt(np.mean((yh - y) ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-8))

N, batch, epochs = 24000, 64, 4
net = Net()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
nper = N // batch
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * nper, 1e-4)
crit = nn.MSELoss()
CIN = torch.from_numpy(np.stack([make_features(z)[0] for z in CN]))
best = (1e9, None)
for ep in range(epochs):
    t0 = time.time()
    idx = rng.integers(0, len(XALL), N)
    Zz, X0, SIG = corrupt_batch(XALL[idx], rng)
    INP = np.stack([make_features(z)[0] for z in Zz])
    tot = 0
    for k in range(nper):
        sl = slice(k * batch, (k + 1) * batch)
        out = net(torch.from_numpy(INP[sl]))
        loss = crit(out, torch.from_numpy(X0[sl].astype(np.float32))[:, None])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        tot += loss.item() * batch
    with torch.no_grad():
        yh = net(CIN).numpy().clip(0, 1)
    v = nrmse(yh[:, 0], CC)
    print(f"epoch {ep}: mse {tot / N:.5f} calib-NRMSE {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if v < best[0]:
        best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})

# calibration fine-tune
net.load_state_dict(best[1])
opt2 = torch.optim.Adam(net.parameters(), lr=3e-5)
CCIN = torch.from_numpy(CC.astype(np.float32))[:, None]
bestv = best[0]; patience = 0
for step in range(120):
    out = net(CIN)
    loss = crit(out, CCIN)
    opt2.zero_grad(); loss.backward(); opt2.step()
    if step % 10 == 9:
        with torch.no_grad():
            yh = net(CIN).numpy().clip(0, 1)
        v = nrmse(yh[:, 0], CC)
        print(f"calib-ft step {step}: calib-NRMSE {v:.4f}", flush=True)
        if v < bestv - 1e-4:
            bestv = v
            best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
net.load_state_dict(best[1])
torch.save(net.state_dict(), '_denoiser_final.pt')
with torch.no_grad():
    yh = net(CIN).numpy().clip(0, 1)
print("FINAL calib-NRMSE %.4f" % nrmse(yh[:, 0], CC))
print("per-image:", np.round([nrmse(yh[i, 0], CC[i]) for i in range(24)], 3))
np.save('_calib_restored.npy', yh[:, 0])
