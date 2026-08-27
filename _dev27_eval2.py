import numpy as np, torch
import _dev26_cls_v2 as m
from _dev20_margin import margins
from _dev24_bayes import verdict_scores

_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
REST = np.load('_rest_arch.npy')
SIGZ = np.load('_sig_arch.npy')
ANCH_IDX, ANCH_LAB = m.ANCH_IDX, m.ANCH_LAB

SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def read(model, F_list):
    outs = []
    with torch.no_grad():
        for F in F_list:
            outs.append(torch.softmax(model(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)


def feats_mixed():
    out = []
    for sy, sx in SHIFTS:
        Rs = np.roll(REST, (sy, sx), (1, 2))
        inp = np.empty((len(REST), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rs; inp[:, 1] = SIGZ[:, None, None]
        out.append(inp)
    return out


def feats_noisy():
    out = []
    for sy, sx in SHIFTS:
        Zs = np.roll(Z, (sy, sx), (1, 2))
        out.append(m.make_features_C_batch(Zs)[0])
    return out


cm = m.Cls2(2); cm.load_state_dict(torch.load('_cls_mixed2.pt')); cm.eval()
cn = m.Cls2(4); cn.load_state_dict(torch.load('_cls_noisy2.pt')); cn.eval()

Pm = read(cm, feats_mixed())
print('mixed2 anchor acc %.4f' % (Pm[ANCH_IDX].argmax(1) == ANCH_LAB).mean(), flush=True)
Pn = read(cn, feats_noisy())
print('noisy2 anchor acc %.4f' % (Pn[ANCH_IDX].argmax(1) == ANCH_LAB).mean(), flush=True)

best = None
for T in np.arange(0.5, 2.01, 0.05):
    Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
    Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
    for w in np.arange(0.0, 1.01, 0.05):
        a = float(((w * Qm + (1 - w) * Qn)[ANCH_IDX].argmax(1) == ANCH_LAB).mean())
        if best is None or a > best[0]:
            best = (a, T, w)
a, T, w = best
print('best blend T=%.2f w=%.2f anchor acc %.4f' % (T, w, a), flush=True)
Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
P = w * Qm + (1 - w) * Qn
np.savez('_probs_v2.npz', Pm=Pm, Pn=Pn, P=P, T=np.array([T]), w=np.array([w]))

vs = verdict_scores(P)                       # list of (r, score, pos, val)
flat = np.full((64, 70), -np.inf)
for r, s, p, v in vs:
    flat[r, p * 10 + v] = s
i = int(np.argmax(flat))
r, pv = divmod(i, 70)
p, v = divmod(pv, 10)
top = np.argsort(flat.max(1))[::-1][:8]
print('bayes verdict: R%02d:%d:%d' % (r, p, v))
print('row margins:', [(int(rr), round(float(flat.max(1)[rr]), 2)) for rr in top])
print('done')
