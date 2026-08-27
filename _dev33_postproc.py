import numpy as np
import torch
from _utils import warp_batch, hblur_batch, make_features_batch, nrmse, corrupt_batch
from _dev15_cls_final import DenNet
from _dev28_den_v3 import make_features_C_batch

torch.set_num_threads(8)
MIX = (0.80, 25.0, 110.0, 0.0, 200.0)

X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
XALL = np.concatenate([X_TRAIN, X_DIG])
_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
CN = _a['calib_noisy'].astype(np.float64)
CC = _a['calib_clean'].astype(np.float64)

da = DenNet(2); da.load_state_dict(torch.load('_den_a.pt')); da.eval()
dc = DenNet(4); dc.load_state_dict(torch.load('_den_c.pt')); dc.eval()


def both(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = da(torch.from_numpy(FA)).numpy()[:, 0]
        RC = dc(torch.from_numpy(FC)).numpy()[:, 0]
    return RA, RC, sig


rng = np.random.default_rng(4242)
NSIM = 800
sidx = rng.integers(0, len(XALL), NSIM)
ZS, X0S, SIGS = corrupt_batch(XALL[sidx], rng, t_mix=MIX)
RA, RC, SIG = both(np.concatenate([Z, ZS]))
RAS, RCS = RA[512:], RC[512:]
RAZ, RCZ = RA[:512], RC[:512]
RAC, RCC, SIGC = both(CN)

print('== ensemble weight sweep (SIM, pooled NRMSE) ==')
for w in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
    print('wA=%.2f : %.4f' % (w, nrmse(np.clip(w * RAS + (1 - w) * RCS, 0, 1), X0S)))

E = np.clip(0.5 * RAS + 0.5 * RCS, 0, 1)


def report(tag, R, X0):
    e = np.sqrt(((R - X0) ** 2).mean((1, 2))) / (np.sqrt((X0 ** 2).mean((1, 2))) + 1e-8)
    print('%s pooled %.4f per-crop %.4f' % (tag, nrmse(R, X0), e.mean()))


print('== post-processing on SIM ==')
report('baseline ens', E, X0S)

# affine fit on sim itself is cheating; fit on CALIB pairs, apply to sim
from numpy.linalg import lstsq
Ec = np.clip(0.5 * RAC + 0.5 * RCC, 0, 1)
A_ = np.stack([Ec.ravel(), np.ones(Ec.size)], 1)
coef, *_ = lstsq(A_, CC.ravel(), rcond=None)
a, b = coef
print('affine fit on calib: a=%.3f b=%.3f' % (a, b))
report('affine(calib-fit)', np.clip(a * E + b, 0, 1), X0S)

# per-crop affine: map each crop's [p5,p95] to [0,1]
def stretch(R, lo=5, hi=95):
    R2 = R.copy()
    for i in range(len(R)):
        l, hh = np.percentile(R[i], lo), np.percentile(R[i], hi)
        if hh - l > 0.05:
            R2[i] = (R[i] - l) / (hh - l)
    return np.clip(R2, 0, 1)
report('quantile stretch', stretch(E), X0S)

# unsharp: E + k*(E - blur3(E))
from scipy.ndimage import uniform_filter
BL = uniform_filter(E, size=(1, 3, 3))
for k in (0.2, 0.4, 0.6):
    report('unsharp k=%.1f' % k, np.clip(E + k * (E - BL), 0, 1), X0S)

# gamma
for g in (0.8, 0.9, 1.1, 1.2):
    report('gamma %.1f' % g, np.clip(E ** g, 0, 1), X0S)

# median filter 3x3
from scipy.ndimage import median_filter
report('median3', median_filter(E, size=(1, 3, 3)), X0S)

print('== calib check ==')
report('calib ens', Ec, CC)
report('calib affine', np.clip(a * Ec + b, 0, 1), CC)
