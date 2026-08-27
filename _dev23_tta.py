import numpy as np, torch
import _utils as u
from _utils import make_features_batch
from _dev15_cls_final import Cls, DenNet, make_features_C_batch, ANCH_IDX, ANCH_LAB
from _dev20_margin import margins

_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
REST = np.load('_rest_arch.npy')
SIGZ = np.load('_sig_arch.npy')

da = DenNet(2); da.load_state_dict(torch.load('_den_a.pt')); da.eval()
dc = DenNet(4); dc.load_state_dict(torch.load('_den_c.pt')); dc.eval()
cm = Cls(2); cm.load_state_dict(torch.load('_cls_mixed_v3ft.pt')); cm.eval()
cn = Cls(4); cn.load_state_dict(torch.load('_cls_noisy_v3ft.pt')); cn.eval()

SH = [(dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]


def read_mixed():
    outs = []
    for sy, sx in SH:
        Rs = np.roll(REST, (sy, sx), (1, 2))
        inp = np.empty((len(REST), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rs; inp[:, 1] = SIGZ[:, None, None]
        with torch.no_grad():
            outs.append(torch.softmax(cm(torch.from_numpy(inp)), 1).numpy())
    return np.mean(outs, 0)


def read_raw():
    outs = []
    for sy, sx in SH:
        Zs = np.roll(Z, (sy, sx), (1, 2))
        F, _ = make_features_C_batch(Zs)
        with torch.no_grad():
            outs.append(torch.softmax(cn(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)


Pm = read_mixed()
print('mixed 25-shift anchor acc %.4f' % (Pm[ANCH_IDX].argmax(1) == ANCH_LAB).mean(), flush=True)
Pn = read_raw()
print('noisy  25-shift anchor acc %.4f' % (Pn[ANCH_IDX].argmax(1) == ANCH_LAB).mean(), flush=True)

best = None
for T in np.arange(0.4, 2.01, 0.1):
    Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
    Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
    for w in np.arange(0.2, 1.01, 0.1):
        P = w * Qm + (1 - w) * Qn
        acc = (P[ANCH_IDX].argmax(1) == ANCH_LAB).mean()
        if best is None or acc > best[0]:
            best = (acc, T, w)
print('best blend T=%.1f w=%.1f acc %.4f' % best, flush=True)
acc, T, w = best
Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
P = w * Qm + (1 - w) * Qn
np.savez('_probs_tta.npz', Pm=Pm, Pn=Pn, P=P)

ms = margins(P)
top = sorted(ms, key=lambda t: -t[1])[:10]
print('margin top:', [(r, round(m, 2), (c[0], c[1])) for r, m, c in top])
pm_ = margins(Pm); pn_ = margins(Pn)
print('margin Pm :', [(r, round(m, 2)) for r, m, c in sorted(pm_, key=lambda t: -t[1])[:6]])
print('margin Pn :', [(r, round(m, 2)) for r, m, c in sorted(pn_, key=lambda t: -t[1])[:6]])
