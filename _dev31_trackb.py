import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from _utils import make_features_batch, nrmse

T0 = time.time()
torch.set_num_threads(8)

WEIGHTS = np.array([7, 3, 1, 7, 3, 1, 7])
WARDS = [1, 2, 3, 4]
FORMULARY = {"05": (10, 40), "08": (5, 25), "11": (20, 60), "17": (15, 45),
             "23": (30, 90), "26": (10, 50), "34": (25, 75), "39": (5, 35),
             "42": (40, 95), "51": (20, 70), "63": (15, 55), "77": (35, 85)}


def check_digit(body):
    return int(np.dot(WEIGHTS, np.asarray(body)) % 10)


def row_is_legal(r, row8):
    d = list(map(int, row8))
    if d[0] not in WARDS: return False
    if d[1] * 10 + d[2] != r + 1: return False
    code = f"{d[3]}{d[4]}"
    if code not in FORMULARY: return False
    lo, hi = FORMULARY[code]
    return lo <= d[5] * 10 + d[6] <= hi


def make_features_C_batch(Zz):
    Zz = np.asarray(Zz, dtype=np.float64)
    B = len(Zz)
    med = median_filter(Zz, size=(1, 3, 3), mode='nearest')
    bad = ((Zz <= -0.349) & (med > -0.10)) | ((Zz >= 1.349) & (med < 0.60))
    ZD = np.where(bad, med, Zz)
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    dv = ZD[:, 1:, :] - ZD[:, :-1, :]
    okd = ~clip0[:, 1:, :] & ~clip0[:, :-1, :]
    est = np.empty(B)
    for i in range(B):
        d = dv[i][okd[i]]
        if len(d) < 200:
            d = dv[i].ravel()
        md = np.median(d)
        est[i] = np.median(np.abs(d - md))
    est = np.clip((est / 0.6745 / np.sqrt(2)) / 0.962, 0.03, 1.2)
    from _utils import SIG_CORRECT
    sig = SIG_CORRECT(est)
    feat = np.empty((B, 4, 28, 28), dtype=np.float32)
    for i in range(B):
        v = ZD[i][~clip0[i]]
        if len(v) >= 200:
            mu = np.percentile(v, 15); hi = np.percentile(v, 97)
            sc = max(hi - mu, 3 * sig[i], 0.25)
        else:
            mu, sc = 0.0, 1.0
        feat[i, 0] = np.clip((ZD[i] - mu) / sc, -1.5, 2.5)
        feat[i, 1] = ZD[i]
        feat[i, 2] = sig[i]
        feat[i, 3] = clip0[i].astype(np.float32)
    return feat, sig


class ResBlk(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.a = nn.Conv2d(w, w, 3, padding=1)
        self.b1 = nn.BatchNorm2d(w)
        self.c = nn.Conv2d(w, w, 3, padding=1)
        self.b2 = nn.BatchNorm2d(w)

    def forward(self, x):
        y = self.b1(torch.relu(self.a(x)))
        y = self.b2(self.c(y))
        return torch.relu(x + y)


class Cls2(nn.Module):
    def __init__(self, nch):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(nch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(2))
        self.lift = nn.Conv2d(64, 128, 1)
        self.s2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                ResBlk(128), nn.MaxPool2d(2))
        self.s3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                ResBlk(256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        self.head = nn.Linear(256, 10)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(self.lift(x))
        x = self.s3(x)
        x = self.pool(x).flatten(1)
        return self.head(self.drop(x))


_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
tr = np.load('_tmpl_rest.npz')
TMPL, QUAL, RREG, SIG = tr['tmpl'][:512], tr['qual'][:512], tr['R'][:512], tr['SIG'][:512]
LAB = tr['lab'][:512]
N_ROWS = 64

mode = sys.argv[1] if len(sys.argv) > 1 else 'ens'
if mode == 'ens':
    restored = RREG.copy()
elif mode == 'tmpl':
    restored = TMPL.copy()
elif mode == 'gate':
    th = float(sys.argv[2])
    restored = RREG.copy()
    g = QUAL >= th
    restored[g] = TMPL[g]
    print('gate %.2f: %d template crops' % (th, g.sum()))
else:
    q0, q1 = float(sys.argv[2]), float(sys.argv[3])
    w = np.clip((QUAL - q0) / (q1 - q0), 0, 1)[:, None, None]
    restored = w * TMPL + (1 - w) * RREG
    print('blend %.2f-%.2f mean w %.3f' % (q0, q1, w.mean()))

CLS_M = Cls2(2); CLS_M.load_state_dict(torch.load('_cls_mixed_v3ft.pt')); CLS_M.eval()
CLS_N = Cls2(4); CLS_N.load_state_dict(torch.load('_cls_noisy_v3ft.pt')); CLS_N.eval()
for m in (CLS_M, CLS_N):
    m.eval()

ANCH_IDX = np.array([8 * r + j for r in range(N_ROWS) for j in (1, 2)])
ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10
                     for r in range(N_ROWS) for j in (1, 2)])
SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def read_restored(R, sig):
    outs = []
    for sy, sx in SHIFTS:
        Rs = np.roll(R, (sy, sx), (1, 2))
        inp = np.empty((len(R), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rs; inp[:, 1] = sig[:, None, None]
        with torch.no_grad():
            outs.append(torch.softmax(CLS_M(torch.from_numpy(inp)), 1).numpy())
    return np.mean(outs, 0)


def read_raw(Zz):
    outs = []
    for sy, sx in SHIFTS:
        Zs = np.roll(Zz, (sy, sx), (1, 2))
        F, _ = make_features_C_batch(Zs)
        with torch.no_grad():
            outs.append(torch.softmax(CLS_N(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)


Pm = read_restored(restored, SIG)
Pn = read_raw(Z)
PT = np.full((512, 10), 0.0)
PT[np.arange(512), LAB] = 1.0
_acc = lambda Qk: float((Qk[ANCH_IDX].argmax(1) == ANCH_LAB).mean())
print('anchor acc: restored-read %.3f | raw-read %.3f | template-label %.3f'
      % (_acc(Pm), _acc(Pn), _acc(PT)))

best = (-1.0, 1.0, 0.5, 0.0)
for T in np.arange(0.5, 2.01, 0.1):
    Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
    Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
    for wgt in np.arange(0.0, 1.01, 0.1):
        B = wgt * Qm + (1 - wgt) * Qn
        for wt in (0.0, 0.1, 0.2, 0.3, 0.4):
            a = _acc((1 - wt) * B + wt * PT)
            if a > best[0]:
                best = (a, T, wgt, wt)
TEMP, W_BLEND, W_TMPL = best[1], best[2], best[3]
Qm = Pm ** (1 / TEMP); Qm /= Qm.sum(1, keepdims=True)
Qn = Pn ** (1 / TEMP); Qn /= Qn.sum(1, keepdims=True)
B = W_BLEND * Qm + (1 - W_BLEND) * Qn
P = (1 - W_TMPL) * B + W_TMPL * PT
print('blend: T=%.2f w=%.2f wt=%.2f -> anchor acc %.4f' % (TEMP, W_BLEND, W_TMPL, _acc(P)))

lg = np.log(np.clip(P, 1e-12, 1))
LB = []
for d0 in WARDS:
    for code, (lo, hi) in FORMULARY.items():
        c0, c1 = int(code[0]), int(code[1])
        for dose in range(lo, hi + 1):
            d5, d6 = dose // 10, dose % 10
            d7v = int((WEIGHTS[0] * d0 + WEIGHTS[3] * c0 + WEIGHTS[4] * c1
                       + WEIGHTS[5] * d5 + WEIGHTS[6] * d6) % 10)
            LB.append((d0, c0, c1, d5, d6, d7v))
LB = np.array(LB)
SLOTS = [0, 3, 4, 5, 6, 7]
CORRUPTIBLE = [0, 3, 4, 5, 6]


def _lse(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def verdict_scores(lp_all):
    out = []
    for r in range(N_ROWS):
        l8 = lp_all[8 * r:8 * r + 8]
        bed = ((r + 1) // 10, (r + 1) % 10)
        base = l8[1, bed[0]] + l8[2, bed[1]]
        lslot = np.stack([l8[s, LB[:, k]] for k, s in enumerate(SLOTS)], 1)
        sfull = base + lslot.sum(1)
        lz0 = _lse(sfull)
        sother = base + lslot.sum(1, keepdims=True) - lslot
        scores = np.full((7, 10), -np.inf)
        for k_, p in enumerate(CORRUPTIBLE):
            s_o = sother[:, k_]
            for v in range(10):
                mask = LB[:, k_] != v
                if mask.any():
                    scores[p, v] = _lse(np.where(mask, s_o + l8[p, v], -np.inf)) - lz0
        out.append(scores)
    return np.array(out)


VS = verdict_scores(lg)
vs_flat = VS.reshape(N_ROWS, -1)
order = np.argsort(-vs_flat.max(1))
print('verdict ranking (row: score):')
for r in order[:6]:
    i_best = int(np.argmax(vs_flat[r]))
    pv, val = divmod(i_best, 10)
    print('  row %2d pos %d written %d score %.2f' % (r, CORRUPTIBLE[pv], val, vs_flat[r].max()))
i_best = int(np.argmax(vs_flat))
r_v, pv = divmod(i_best, 70)
pos_v, val_v = divmod(pv, 10)
l8v = lg[8 * r_v:8 * r_v + 8]
bed = ((r_v + 1) // 10, (r_v + 1) % 10)
lslot = np.stack([l8v[s, LB[:, k]] for k, s in enumerate(SLOTS)], 1)
base = l8v[1, bed[0]] + l8v[2, bed[1]]
s_o = base + lslot.sum(1) - lslot[:, CORRUPTIBLE.index(pos_v)]
i_L = int(np.argmax(s_o))
true_v = int(LB[i_L, CORRUPTIBLE.index(pos_v)])
verdict = (r_v, pos_v, true_v)
ledger = 'R%02d:%d:%d' % verdict
print('VERDICT %s (written %d at pos %d, should be %d, near-tie rows within 1 nat: %d)'
      % (ledger, val_v, pos_v, true_v,
         int((vs_flat.max(1) > vs_flat.max() - 1.0).sum())))


def best_legal_parse(r, lp):
    bed = r + 1
    d1, d2 = bed // 10, bed % 10
    bestp = (-np.inf, None)
    for d0 in WARDS:
        s0 = lp[0, d0]
        for code, (lo, hi) in FORMULARY.items():
            c0, c1 = int(code[0]), int(code[1])
            s01 = s0 + lp[3, c0] + lp[4, c1]
            for dose in range(lo, hi + 1):
                d5, d6 = dose // 10, dose % 10
                body = [d0, d1, d2, c0, c1, d5, d6]
                d7 = check_digit(body)
                sc = s01 + lp[5, d5] + lp[6, d6] + lp[7, d7]
                if sc > bestp[0]:
                    bestp = (float(sc), body + [d7])
    return bestp


digits = P.argmax(1).copy()
fails, nat_of = [], {}
for r in range(N_ROWS):
    p8, lp8 = P[8 * r:8 * r + 8], lg[8 * r:8 * r + 8]
    nat = [int(np.array(WARDS)[np.argmax(p8[0, WARDS])]), (r + 1) // 10, (r + 1) % 10] \
        + [int(p8[j].argmax()) for j in range(3, 8)]
    nat_of[r] = nat
    if not (row_is_legal(r, nat) and check_digit(nat[:7]) == nat[7]):
        sc_legal, legal_row = best_legal_parse(r, lp8)
        fails.append(dict(r=r, nat=nat, legal_row=legal_row))
for r in range(N_ROWS):
    digits[8 * r + 1] = (r + 1) // 10
    digits[8 * r + 2] = (r + 1) % 10
    digits[8 * r + 0] = nat_of[r][0]
for f in fails:
    r = f['r']
    if r == verdict[0]:
        for j in range(8):
            digits[8 * r + j] = f['nat'][j]
        digits[8 * r + 1] = (r + 1) // 10
        digits[8 * r + 2] = (r + 1) % 10
    elif f['legal_row'] is not None:
        for j in range(8):
            digits[8 * r + j] = f['legal_row'][j]
print('%d rows fail natural read' % len(fails))

prev = np.load('submission.npz', allow_pickle=True)
ndiff = int((digits != prev['digits']).sum())
print('digit diffs vs previous submission: %d ; prev ledger %s -> new %s'
      % (ndiff, str(prev['ledger']), ledger))
np.savez_compressed('_final_trackb.npz', restored=restored.astype(np.float32),
                    digits=digits, ledger=ledger, P=P)
print('saved _final_trackb.npz in %.0fs' % (time.time() - T0), flush=True)
