import numpy as np, torch, itertools
import _utils as u
from _utils import make_features_batch, nrmse
import _dev9_decide as dec
from _dev15_cls_final import Cls, DenNet, make_features_C_batch, ANCH_IDX, ANCH_LAB

_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)

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
np.save('_rest_arch.npy', REST)
np.save('_sig_arch.npy', SIGZ)

SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def logits_mixed(cm, R, sig):
    outs = []
    for sy, sx in SHIFTS:
        Rs = np.roll(R, (sy, sx), (1, 2))
        inp = np.empty((len(R), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rs; inp[:, 1] = sig[:, None, None]
        with torch.no_grad():
            outs.append(torch.softmax(cm(torch.from_numpy(inp)), 1).numpy())
    return np.mean(outs, 0)


def logits_noisy(cn4, Zz):
    outs = []
    for sy, sx in SHIFTS:
        Zs = np.roll(Zz, (sy, sx), (1, 2))
        F, _ = make_features_C_batch(Zs)
        with torch.no_grad():
            outs.append(torch.softmax(cn4(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)


PMS, PNS = {}, {}
for tag in ('', 'ft'):
    cm = Cls(2); cm.load_state_dict(torch.load(f'_cls_mixed_v3{tag}.pt')); cm.eval()
    cn = Cls(4); cn.load_state_dict(torch.load(f'_cls_noisy_v3{tag}.pt')); cn.eval()
    PMS[tag] = logits_mixed(cm, REST, SIGZ)
    PNS[tag] = logits_noisy(cn, Z)
    name = tag or 'pre'
    am = (PMS[tag][ANCH_IDX].argmax(1) == ANCH_LAB).mean()
    an = (PNS[tag][ANCH_IDX].argmax(1) == ANCH_LAB).mean()
    print(f"{name}: anchor acc mixed {am:.4f} noisy {an:.4f}", flush=True)

best = None
for tm, tn in itertools.product(('', 'ft'), ('', 'ft')):
    for T in np.arange(0.5, 2.01, 0.1):
        Qm = PMS[tm] ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
        Qn = PNS[tn] ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
        for w in np.arange(0.3, 1.01, 0.1):
            P = w * Qm + (1 - w) * Qn
            acc = (P[ANCH_IDX].argmax(1) == ANCH_LAB).mean()
            if best is None or acc > best[0]:
                best = (acc, tm, tn, T, w)
acc, tm, tn, T, w = best
print('best: mixed=%s noisy=%s T=%.2f w=%.2f anchor acc %.4f'
      % (tm or 'pre', tn or 'pre', T, w, acc))
Qm = PMS[tm] ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
Qn = PNS[tn] ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
P = w * Qm + (1 - w) * Qn
digits, ledger, fails, nat_of = dec.adjudicate(P)
np.savez('_stage3.npz', REST=REST, P=P, Pm=PMS[tm], Pn=PNS[tn], digits=digits)
open('_ledger.txt', 'w').write(ledger)
print('ledger:', ledger)
