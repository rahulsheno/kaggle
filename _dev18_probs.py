import numpy as np, torch
from _utils import make_features_batch
from _dev15_cls_final import Cls, make_features_C_batch, ANCH_IDX, ANCH_LAB

_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
REST = np.load('_rest_arch.npy')
SIGZ = np.load('_sig_arch.npy')
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


out = {}
for tag in ('', 'ft'):
    cm = Cls(2); cm.load_state_dict(torch.load(f'_cls_mixed_v3{tag}.pt')); cm.eval()
    cn = Cls(4); cn.load_state_dict(torch.load(f'_cls_noisy_v3{tag}.pt')); cn.eval()
    out['Pm' + tag] = logits_mixed(cm, REST, SIGZ)
    out['Pn' + tag] = logits_noisy(cn, Z)
    print(f'{tag or "pre"} done', flush=True)
np.savez('_probs_all.npz', **out)
print('saved _probs_all.npz')
