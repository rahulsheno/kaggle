import numpy as np, torch
import _utils as u
from _utils import warp_batch, corrupt_batch, make_features_batch
from _dev15_cls_final import (Cls, DenNet, make_features_C_batch,
                              ANCH_IDX, ANCH_LAB, MIX)

da = DenNet(2); da.load_state_dict(torch.load('_den_a.pt')); da.eval()
dc = DenNet(4); dc.load_state_dict(torch.load('_den_c.pt')); dc.eval()
cm = Cls(2); cm.load_state_dict(torch.load('_cls_mixed_v3.pt')); cm.eval()


def restore(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = da(torch.from_numpy(FA)).numpy()[:, 0]
        RC = dc(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig


X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
Y_TRAIN = np.load('y_kannada_MNIST_train.npz')['arr_0'].astype(np.int64)
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
Y_DIG = np.load('y_dig_MNIST.npz')['arr_0'].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG]); YALL = np.concatenate([Y_TRAIN, Y_DIG])

N = 4000
rng = np.random.default_rng(3)
idx = rng.integers(0, len(XALL), N)
X0 = np.clip(warp_batch(XALL[idx], rng.uniform(-15, 15, N), rng.uniform(0.85, 1.15, N),
                        rng.uniform(-2.5, 2.5, N), rng.uniform(-2.5, 2.5, N)), 0, 1)
Zz, _, SIGT = corrupt_batch(X0, rng, t_mix=MIX)
R, SIGE = restore(Zz)
inp = np.empty((N, 2, 28, 28), dtype=np.float32)
inp[:, 0] = R; inp[:, 1] = SIGE[:, None, None]
with torch.no_grad():
    P = torch.softmax(cm(torch.from_numpy(inp)), 1).numpy()
acc = (P.argmax(1) == YALL[idx]).mean()
print('SIM overall acc %.4f' % acc)
for lo, hi in ((0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.8), (0.8, 1.3)):
    m = (SIGE >= lo) & (SIGE < hi)
    if m.sum():
        print('  sim sig [%.2f,%.2f): n=%d acc %.3f' % (lo, hi, m.sum(), (P[m].argmax(1) == YALL[idx][m]).mean()))

# real archive anchors, same model
_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
RA, SGA = restore(Z[ANCH_IDX])
inp2 = np.empty((len(ANCH_IDX), 2, 28, 28), dtype=np.float32)
inp2[:, 0] = RA; inp2[:, 1] = SGA[:, None, None]
with torch.no_grad():
    PA = torch.softmax(cm(torch.from_numpy(inp2)), 1).numpy()
print('REAL anchor acc %.4f (pre-ft mixed)' % (PA.argmax(1) == ANCH_LAB).mean())
for lo, hi in ((0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.8), (0.8, 1.3)):
    m = (SGA >= lo) & (SGA < hi)
    if m.sum():
        print('  real sig [%.2f,%.2f): n=%d acc %.3f' % (lo, hi, m.sum(), (PA[m].argmax(1) == ANCH_LAB[m]).mean()))

# confusion on real anchors
cmf = np.zeros((10, 10), int)
for t, p in zip(ANCH_LAB, PA.argmax(1)):
    cmf[t, p] += 1
print('confusion (rows=true):')
print(cmf)
