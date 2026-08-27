import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import _utils as u
from _utils import make_features_batch, nrmse
import _dev9_decide as dec

X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64); CN = _a["calib_noisy"].astype(np.float64); CC = _a["calib_clean"].astype(np.float64)

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

den_files = [('_denoiser_final.pt', 2)]
import os, sys
if os.path.exists('_den_a.pt') and os.path.exists('_den_c.pt'):
    den_files = [('_den_a.pt', 2), ('_den_c.pt', 4)]

RESTS = []
for fn, nch in den_files:
    den = DenNet(nch, 56, 7)
    den.load_state_dict(torch.load(fn))
    den.eval()
    if nch == 2:
        Fz, SIGZ = make_features_batch(Z)
        FC, _ = make_features_batch(CN)
    else:
        from _dev10_den2 import make_features_C_batch
        Fz, SIGZ = make_features_C_batch(Z)
        FC, _ = make_features_C_batch(CN)
    with torch.no_grad():
        RESTS.append(den(torch.from_numpy(Fz)).numpy()[:, 0].clip(0, 1))
        RC = den(torch.from_numpy(FC)).numpy()[:, 0].clip(0, 1)
    print(f"{fn}: calib NRMSE {nrmse(RC, CC):.4f}")
REST = np.mean(RESTS, 0)
print("archive sigma estimates: med %.3f p90 %.3f max %.3f" % (np.median(SIGZ), np.percentile(SIGZ, 90), SIGZ.max()))

cm = Cls(2); cm.load_state_dict(torch.load('_cls_mixed.pt')); cm.eval()
cn4 = Cls(4); cn4.load_state_dict(torch.load('_cls_noisy4.pt')); cn4.eval()

def feats_noisy4(Zz):
    F, sig = make_features_batch(Zz)
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    ZD = F[:, 0].astype(np.float64)
    feat = np.empty((len(Zz), 4, 28, 28), dtype=np.float32)
    for i in range(len(Zz)):
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
    return feat

INP_M = np.empty((len(Z), 2, 28, 28), dtype=np.float32)
INP_M[:, 0] = REST
INP_M[:, 1] = SIGZ[:, None, None]
INP_N = feats_noisy4(Z)

def logits(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(net(torch.from_numpy(X[i:i + 128])).numpy())
    return np.concatenate(out)

lg_mixed = logits(cm, INP_M)
lg_noisy = logits(cn4, INP_N)

anchor_idx = np.array([8 * r + j for r in range(64) for j in (1, 2)])
anchor_lab = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10 for r in range(64) for j in (1, 2)])

for name, lg in [('mixed', lg_mixed), ('noisy4', lg_noisy)]:
    p = torch.softmax(torch.from_numpy(lg), 1).numpy()
    print(f"anchor acc {name}: {(p[anchor_idx].argmax(1) == anchor_lab).mean():.4f}")

best_T, best_acc, best_w = 1.0, -1, 0.5
for T in np.arange(0.4, 2.21, 0.1):
    Pm = torch.softmax(torch.from_numpy(lg_mixed) / T, 1).numpy()
    Pn = torch.softmax(torch.from_numpy(lg_noisy) / T, 1).numpy()
    for w in np.arange(0.3, 0.91, 0.1):
        P = w * Pm + (1 - w) * Pn
        acc = (P[anchor_idx].argmax(1) == anchor_lab).mean()
        if acc > best_acc:
            best_acc, best_T, best_w = acc, T, w
print("best T=%.2f w=%.2f -> anchor acc %.4f" % (best_T, best_w, best_acc))

Pm = torch.softmax(torch.from_numpy(lg_mixed) / best_T, 1).numpy()
Pn = torch.softmax(torch.from_numpy(lg_noisy) / best_T, 1).numpy()
P = best_w * Pm + (1 - best_w) * Pn

digits, ledger, fails, nat_of = dec.adjudicate(P)
np.savez('_stage2.npz', REST=REST, P=P, digits=digits)
open('_ledger.txt', 'w').write(ledger)
bad = 0
for r in range(64):
    row = digits[8 * r: 8 * r + 8].tolist()
    if not (row[1] * 10 + row[2] == r + 1 and dec.check_digit(row[:7]) == row[7]):
        bad += 1
print(f"rows still failing check with final digits: {bad}")
