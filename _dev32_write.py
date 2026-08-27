import re
import sys
import numpy as np

d = np.load('_final_trackb.npz', allow_pickle=True)
R = np.asarray(d['restored'], dtype=np.float32)
D = np.asarray(d['digits']).astype(np.int64).reshape(-1)
L = str(d['ledger']).strip().upper()
assert R.shape == (512, 28, 28) and np.isfinite(R).all()
assert D.shape == (512,) and 0 <= D.min() and D.max() <= 9
m = re.fullmatch(r"R(\d{2}):([0-6]):(\d)", L)
assert m, L
R = np.clip(R, 0, 1)
np.savez_compressed('submission.npz', restored=R, digits=D, ledger=L)
print('wrote submission.npz ledger', L)
lrow, lpos, ldig = int(m.group(1)), int(m.group(2)), int(m.group(3))
lines = ['Id,Value']
for c in range(512):
    px = R[c].ravel(order='F')
    lines += ['P_%03d_%03d,%.6f' % (c, k, v) for k, v in enumerate(px)]
lines += ['D_%03d,%d' % (c, v) for c, v in enumerate(D)]
lines += ['L_ROW,%d' % lrow, 'L_POS,%d' % lpos, 'L_DIG,%d' % ldig,
          'LP_ROW,%d' % lrow, 'LP_POS,%d' % lpos, 'LP_DIG,%d' % ldig]
with open('submission.csv', 'w', newline='') as f:
    f.write('\n'.join(lines) + '\n')
assert len(lines) == 1 + 512 * 784 + 512 + 6
print('wrote submission.csv (%d data rows)' % (len(lines) - 1))
print('restored mean %.4f  digits hist %s' % (R.mean(), np.bincount(D, minlength=10)))
