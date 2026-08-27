import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import _utils as u
from _utils import make_features_batch, nrmse
import _dev9_decide as dec

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

X_TRAIN = np.load("X_kannada_MNIST_train.npz")["arr_0"].astype(np.float64) / 255.0
Y_TRAIN = np.load("y_kannada_MNIST_train.npz")["arr_0"].astype(np.int64)
_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64); CN = _a["calib_noisy"].astype(np.float64); CC = _a["calib_clean"].astype(np.float64)

den = DenNet(2, 56, 7)
den.load_state_dict(torch.load('_denoiser_final.pt'))
den.eval()
Fz, SIGZ = make_features_batch(Z)
with torch.no_grad():
    REST = den(torch.from_numpy(Fz)).numpy()[:, 0].clip(0, 1)
FC, _ = make_features_batch(CN)
with torch.no_grad():
    RC = den(torch.from_numpy(FC)).numpy()[:, 0].clip(0, 1)
print("calib NRMSE of denoiser:", nrmse(RC, CC))
print("archive sigma estimates: med %.3f  p90 %.3f  max %.3f" % (np.median(SIGZ), np.percentile(SIGZ, 90), SIGZ.max()))

cc = Cls(1); cc.load_state_dict(torch.load('_cls_clean.pt')); cc.eval()
cn = Cls(2); cn.load_state_dict(torch.load('_cls_noisy.pt')); cn.eval()

def logits(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(net(torch.from_numpy(X[i:i + 128].astype(np.float32))).numpy())
    return np.concatenate(out)

lg_clean = logits(cc, REST[:, None])
lg_noisy = logits(cn, Fz)

anchor_idx = np.array([8 * r + j for r in range(64) for j in (1, 2)])
anchor_lab = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10 for r in range(64) for j in (1, 2)])

for name, lg in [('clean', lg_clean), ('noisy', lg_noisy)]:
    p = torch.softmax(torch.from_numpy(lg), 1).numpy()
    print(f"anchor acc {name}: {(p[anchor_idx].argmax(1) == anchor_lab).mean():.4f}")

# temperature tuning on anchors for combined view
best_T, best_acc = 1.0, -1
for T in np.arange(0.4, 2.41, 0.1):
    Pc = torch.softmax(torch.from_numpy(lg_clean) / T, 1).numpy()
    Pn = torch.softmax(torch.from_numpy(lg_noisy) / T, 1).numpy()
    P = (Pc + Pn) / 2
    acc = (P[anchor_idx].argmax(1) == anchor_lab).mean()
    if acc > best_acc:
        best_acc, best_T = acc, T
print("best temperature %.2f -> anchor acc %.4f" % (best_T, best_acc))

Pc = torch.softmax(torch.from_numpy(lg_clean) / best_T, 1).numpy()
Pn = torch.softmax(torch.from_numpy(lg_noisy) / best_T, 1).numpy()
P = (Pc + Pn) / 2
print("combined anchor acc:", (P[anchor_idx].argmax(1) == anchor_lab).mean())

digits, ledger, fails, nat_of = dec.adjudicate(P)
print("saving stage2")
np.savez('_stage2.npz', REST=REST, P=P, digits=digits)
open('_ledger.txt', 'w').write(ledger)

# consistency: re-check all rows with final digits
bad = 0
for r in range(64):
    row = digits[8 * r: 8 * r + 8].tolist()
    ok = row[1] * 10 + row[2] == r + 1 and dec.check_digit(row[:7]) == row[7]
    if not ok:
        bad += 1
print(f"rows still failing check with final digits: {bad}")
