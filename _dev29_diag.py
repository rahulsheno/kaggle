import numpy as np
import torch
import _utils as u
from _utils import corrupt_batch, make_features_batch, nrmse
from _dev15_cls_final import DenNet
from _dev28_den_v3 import make_features_C_batch

torch.set_num_threads(8)

X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
XALL = np.concatenate([X_TRAIN, X_DIG])
_a = np.load('arogya_archive_v1.npz')
CN = _a['calib_noisy'].astype(np.float64)
CC = _a['calib_clean'].astype(np.float64)
MIX = (0.80, 25.0, 110.0, 0.0, 200.0)

da = DenNet(2); da.load_state_dict(torch.load('_den_a.pt')); da.eval()
dc = DenNet(4); dc.load_state_dict(torch.load('_den_c.pt')); dc.eval()


def ens(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = da(torch.from_numpy(FA)).numpy()[:, 0]
        RC = dc(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig


def percrop_nrmse(R, X0):
    e = np.sqrt(((R - X0) ** 2).mean((1, 2))) / (np.sqrt((X0 ** 2).mean((1, 2))) + 1e-8)
    return e


def report(tag, R, X0, sig):
    e = percrop_nrmse(R, X0)
    ez = percrop_nrmse(np.clip(np.asarray([_ for _ in R]), 0, 1) * 0, X0)
    print('%s: pooled %.4f | per-crop mean %.4f (zero=%.3f) | median %.3f p90 %.3f max %.3f'
          % (tag, nrmse(R, X0), e.mean(), ez.mean(), np.median(e),
             np.percentile(e, 90), e.max()))
    for lo, hi in ((0, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.3)):
        m = (sig >= lo) & (sig < hi)
        if m.sum():
            print('  sig[%.2f,%.2f) n=%d pooled %.3f per-crop %.3f'
                  % (lo, hi, m.sum(), nrmse(R[m], X0[m]), e[m].mean()))


Rc, sigc = ens(CN)
report('CALIB ens', Rc, CC, sigc)

rng = np.random.default_rng(1234)
n = 1200
idx = rng.integers(0, len(XALL), n)
Zz, X0, SIG = corrupt_batch(XALL[idx], rng, t_mix=MIX)
Rs, _ = ens(Zz)
report('SIM   ens', Rs, X0, SIG)

ZD = make_features_batch(np.load('arogya_archive_v1.npz')['Z'].astype(np.float64))[0][:, 0]
print('archive desalted: mean %.4f frac>0.5 %.4f' % (ZD.mean(), (ZD > 0.5).mean()))
