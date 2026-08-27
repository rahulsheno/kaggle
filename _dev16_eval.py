import numpy as np, torch
import torch.nn as nn
import _utils as u
from _utils import make_features_batch, nrmse
import _dev9_decide as dec
from _dev15_cls_final import Cls, DenNet, make_features_C_batch, ANCH_IDX, ANCH_LAB

_a = np.load("arogya_archive_v1.npz")
Z = _a["Z"].astype(np.float64); CN = _a["calib_noisy"].astype(np.float64); CC = _a["calib_clean"].astype(np.float64)

da = DenNet(2); da.load_state_dict(torch.load('_den_a.pt')); da.eval()
dc = DenNet(4); dc.load_state_dict(torch.load('_den_c.pt')); dc.eval()

def restore(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = da(torch.from_numpy(FA)).numpy()[:, 0]
        RC = dc(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig

REST, SIGZ = restore(Z)
RCAL, _ = restore(CN)
print("ensemble calib NRMSE:", nrmse(RCAL, CC))
print("archive sigma: med %.3f p90 %.3f max %.3f" % (np.median(SIGZ), np.percentile(SIGZ, 90), SIGZ.max()))

cm = Cls(2); cm.load_state_dict(torch.load('_cls_mixed_v3ft.pt')); cm.eval()
cn4 = Cls(4); cn4.load_state_dict(torch.load('_cls_noisy_v3ft.pt')); cn4.eval()

SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

def logits_mixed(R, sig):
    outs = []
    for sy, sx in SHIFTS:
        Rs = np.roll(R, (sy, sx), (1, 2))
        inp = np.empty((len(R), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rs; inp[:, 1] = sig[:, None, None]
        with torch.no_grad():
            outs.append(torch.softmax(cm(torch.from_numpy(inp)), 1).numpy())
    return np.mean(outs, 0)

def logits_noisy(Zz):
    outs = []
    for sy, sx in SHIFTS:
        Zs = np.roll(Zz, (sy, sx), (1, 2))
        F, _ = make_features_C_batch(Zs)
        with torch.no_grad():
            outs.append(torch.softmax(cn4(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)

Pm = logits_mixed(REST, SIGZ)
Pn = logits_noisy(Z)

for name, Pk in [('mixed-ft', Pm), ('noisy4-ft', Pn)]:
    print(f"anchor acc {name}: {(Pk[ANCH_IDX].argmax(1) == ANCH_LAB).mean():.4f}")

best_T, best_acc, best_w = 1.0, -1, 0.5
for T in np.arange(0.5, 2.01, 0.05):
    Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
    Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
    for w in np.arange(0.3, 0.91, 0.05):
        P = w * Qm + (1 - w) * Qn
        acc = (P[ANCH_IDX].argmax(1) == ANCH_LAB).mean()
        if acc > best_acc:
            best_acc, best_T, best_w = acc, T, w
print("best T=%.2f w=%.2f -> anchor acc %.4f" % (best_T, best_w, best_acc))

Qm = Pm ** (1 / best_T); Qm /= Qm.sum(1, keepdims=True)
Qn = Pn ** (1 / best_T); Qn /= Qn.sum(1, keepdims=True)
P = best_w * Qm + (1 - best_w) * Qn

digits, ledger, fails, nat_of = dec.adjudicate(P)
np.savez('_stage3.npz', REST=REST, P=P, Pm=Pm, Pn=Pn, digits=digits)
open('_ledger.txt', 'w').write(ledger)
print("saved stage3; ledger =", ledger)
