# ==================== STEP 2 - READ THE LEDGER ====================
# Self-contained restoration + reading + adjudication.
# Everything the models know was learned from the official corpus run through the
# GIVEN chain, plus the 24 calibration pairs; nothing in here peeks at labels it
# is not entitled to. Runs on CPU in a few minutes.
import base64, io, time
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter

torch.set_num_threads(max(1, (torch.get_num_threads())))
_T0 = time.time()

# ---- per-crop noise estimation ----------------------------------------------
# Robust MAD of vertical differences on the de-salted image, mapped to true sigma
# with a table fitted by running the GIVEN chain over the official corpus
# (clipping at +/-0.35/1.35 truncates the tails, so the raw MAD under-reads at
# high noise; the table is the inverse of that bias).
BM = np.array([0.061860, 0.094746, 0.106527, 0.140288, 0.163624, 0.185284,
               0.213627, 0.237071, 0.262780, 0.286812, 0.314371, 0.337975,
               0.361558, 0.388068, 0.413752, 0.437743, 0.462849, 0.487541,
               0.511303, 0.535473, 0.561014, 0.583831])
BT = np.array([0.049871, 0.089402, 0.089402, 0.128387, 0.128387, 0.167055,
               0.205415, 0.243436, 0.281070, 0.281070, 0.354974, 0.391137,
               0.426702, 0.495828, 0.593735, 0.683532, 0.834029, 0.892681,
               0.939149, 0.925056, 0.963039, 0.981085])


def make_features_A(Zz):
    """view A: [de-salted z, sigma]"""
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
    sig = np.interp(est, BM, BT, left=0.0, right=BT[-1])
    feat = np.empty((B, 2, 28, 28), dtype=np.float32)
    feat[:, 0] = ZD
    feat[:, 1] = sig[:, None, None]
    return feat, sig


def make_features_C(Zz):
    """view C: [contrast-normalized z, de-salted z, sigma, clipmask]"""
    F_A, sig = make_features_A(Zz)
    Zz = np.asarray(Zz, dtype=np.float64)
    clip0 = (Zz <= -0.349) | (Zz >= 1.349)
    ZD = F_A[:, 0].astype(np.float64)
    feat = np.empty((len(Zz), 4, 28, 28), dtype=np.float32)
    for i in range(len(Zz)):
        v = ZD[i][~clip0[i]]
        if len(v) >= 200:
            mu = np.percentile(v, 15)
            hi = np.percentile(v, 97)
            sc = max(hi - mu, 3 * sig[i], 0.25)
        else:
            mu, sc = 0.0, 1.0
        feat[i, 0] = np.clip((ZD[i] - mu) / sc, -1.5, 2.5)
        feat[i, 1] = ZD[i]
        feat[i, 2] = sig[i]
        feat[i, 3] = clip0[i].astype(np.float32)
    return feat, sig


# ---- models ------------------------------------------------------------------
class DenNet(nn.Module):
    def __init__(self, nch, w=56, nblk=7):
        super().__init__()
        layers = [nn.Conv2d(nch, w, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(w, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x[:, :1] + self.body(x)


class Cls(nn.Module):
    def __init__(self, nch=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(nch, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(192, 10)

    def forward(self, x):
        return self.head(self.f(x))


W_DEN_A = "__W_DEN_A__"
W_DEN_C = "__W_DEN_C__"
W_CLS_MIXED = "__W_CLS_MIXED__"
W_CLS_NOISY = "__W_CLS_NOISY__"


def _load(payload):
    with io.BytesIO(base64.b64decode(payload)) as f:
        return torch.load(f, map_location="cpu", weights_only=True)


den_a = DenNet(2); den_a.load_state_dict(_load(W_DEN_A)); den_a.eval()
den_c = DenNet(4); den_c.load_state_dict(_load(W_DEN_C)); den_c.eval()
cls_m = Cls(2); cls_m.load_state_dict(_load(W_CLS_MIXED)); cls_m.eval()
cls_n = Cls(4); cls_n.load_state_dict(_load(W_CLS_NOISY)); cls_n.eval()
for m in (den_a, den_c, cls_m, cls_n):
    for p in m.parameters():
        p.requires_grad = False


# ---- Track A: restoration ------------------------------------------------------
def restore(Zz):
    FA, sig = make_features_A(Zz)
    FC, _ = make_features_C(Zz)
    with torch.no_grad():
        RA = den_a(torch.from_numpy(FA)).numpy()[:, 0]
        RC = den_c(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig


restored, SIGZ = restore(Z)
RCAL, _ = restore(CALIB_NOISY)
_nrmse = lambda yh, y: float(np.sqrt(np.mean((np.asarray(yh) - np.asarray(y)) ** 2))
                             / (np.sqrt(np.mean(np.asarray(y) ** 2)) + 1e-8))
print("restoration done in %.0fs | calibration NRMSE %.4f (all-zero baseline 1.000)"
      % (time.time() - _T0, _nrmse(RCAL, CALIB_CLEAN)))
print("archive sigma: med %.3f  p90 %.3f  max %.3f"
      % (np.median(SIGZ), np.percentile(SIGZ, 90), SIGZ.max()))
print("restored range [%.3f, %.3f] mean %.4f"
      % (restored.min(), restored.max(), restored.mean()))


# ---- Track B: two independent reads -------------------------------------------
# read 1: on the restored images (restored + sigma channel)
# read 2: straight on the raw archive (contrast-normalized 4-channel view)
# each is averaged over small shift augmentations, then the two are sharpened to a
# common temperature and blended. The blend weights and temperature were chosen on
# the 128 bed-anchor digits, whose true values are fixed by the bed numbering.
def _read_restored(R, sig):
    outs = []
    for sy, sx in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        Rs = np.roll(R, (sy, sx), (1, 2))
        inp = np.empty((len(R), 2, 28, 28), dtype=np.float32)
        inp[:, 0] = Rs
        inp[:, 1] = sig[:, None, None]
        with torch.no_grad():
            outs.append(torch.softmax(cls_m(torch.from_numpy(inp)), 1).numpy())
    return np.mean(outs, 0)


def _read_raw(Zz):
    outs = []
    for sy, sx in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        Zs = np.roll(Zz, (sy, sx), (1, 2))
        F, _ = make_features_C(Zs)
        with torch.no_grad():
            outs.append(torch.softmax(cls_n(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)


_Pm = _read_restored(restored, SIGZ)
_Pn = _read_raw(Z)
TEMP, W_BLEND = __TEMP__, __WBLEND__
Qm = _Pm ** (1.0 / TEMP); Qm /= Qm.sum(1, keepdims=True)
Qn = _Pn ** (1.0 / TEMP); Qn /= Qn.sum(1, keepdims=True)
P = W_BLEND * Qm + (1.0 - W_BLEND) * Qn

ANCH_IDX = np.array([8 * r + j for r in range(N_ROWS) for j in (1, 2)])
ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10
                     for r in range(N_ROWS) for j in (1, 2)])
_acc = lambda Qk: float((Qk[ANCH_IDX].argmax(1) == ANCH_LAB).mean())
print("anchor accuracy: restored-read %.3f | raw-read %.3f | blend %.3f"
      % (_acc(_Pm), _acc(_Pn), _acc(P)))


# ---- adjudication: the ledger is not noisy ------------------------------------
# natural read per row; rows that are both legal (ward/bed/drug/dose) and whose
# check digit closes are accepted. Every failed row is repaired; the verdict is
# the single mis-written digit whose correction makes its row close and stay legal.
def _solve_repair(nat, pos, rhs):
    INV = {1: 1, 3: 7, 7: 3, 9: 9}
    body = nat[:7].copy()
    s = int(sum(WEIGHTS[j] * body[j] for j in range(7) if j != pos))
    return ((rhs - s) % 10) * INV[WEIGHTS[pos]] % 10


def _main_repairs(r, nat):
    d7 = nat[7]
    reps = []
    for pos in (0, 3, 4, 5, 6):
        v = _solve_repair(nat, pos, d7)
        if v == nat[pos]:
            continue
        body = nat[:7].copy(); body[pos] = v
        if pos == 0 and v not in WARDS:
            continue
        if row_is_legal(r, body + [d7]):
            reps.append((pos, v))
    return reps


def _gen_repairs(r, nat, lp, pmin=-4.0):
    reps = []
    for pos in (0, 3, 4, 5, 6):
        for d7v in range(10):
            if lp[7, d7v] < pmin:
                continue
            v = _solve_repair(nat, pos, d7v)
            if v == nat[pos]:
                continue
            body = nat[:7].copy(); body[pos] = v
            if pos == 0 and v not in WARDS:
                continue
            if row_is_legal(r, body + [d7v]):
                score = sum(lp[j, nat[j]] for j in range(7) if j != pos) + lp[7, d7v]
                reps.append((pos, v, d7v, float(score)))
    return reps


def _best_legal_parse(r, lp):
    bed = r + 1
    d1, d2 = bed // 10, bed % 10
    best = (-np.inf, None)
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
                if sc > best[0]:
                    best = (float(sc), body + [d7])
    return best


lg = np.log(np.clip(P, 1e-12, 1))
digits = P.argmax(1).copy()
fails, nat_of = [], {}
for r in range(N_ROWS):
    p8, lp8 = P[8 * r:8 * r + 8], lg[8 * r:8 * r + 8]
    bed = r + 1
    nat = [int(np.array([1, 2, 3, 4])[np.argmax(p8[0, [1, 2, 3, 4]])]),
           bed // 10, bed % 10] + [int(p8[j].argmax()) for j in range(3, 8)]
    nat_of[r] = nat
    legal = row_is_legal(r, nat)
    closes = check_digit(nat[:7]) == nat[7]
    if not (legal and closes):
        mr = _main_repairs(r, nat)
        gr = [] if mr else _gen_repairs(r, nat, lp8)
        sc_legal, legal_row = _best_legal_parse(r, lp8)
        fails.append(dict(r=r, nat=nat, mr=mr, gr=gr,
                          logconf=float(sum(lp8[j, nat[j]] for j in range(8))),
                          tension=sc_legal - float(sum(lp8[j, nat[j]] for j in range(8))),
                          legal_row=legal_row))
print("%d of %d rows fail the natural read" % (len(fails), N_ROWS))

t1 = [f for f in fails if len(f['mr']) == 1]
if t1:
    f = max(t1, key=lambda h: h['logconf'])
    verdict = (f['r'],) + f['mr'][0]
    why = "unique single-digit repair, best log-confidence among %d candidates" % len(t1)
else:
    t2 = [f for f in fails if f['mr']]
    pool = t2 if t2 else [f for f in fails if f['gr']]
    if pool:
        if t2:
            f = max(pool, key=lambda h: h['logconf'])
            verdict = (f['r'],) + f['mr'][0]
            why = "multi-repair row, highest log-confidence"
        else:
            f = max(pool, key=lambda h: max(g[3] for g in h['gr']))
            g = max(f['gr'], key=lambda h: h[3])
            verdict = (f['r'], g[0], g[1])
            why = "generalized repair (check digit remapped to %d)" % g[2]
    else:
        f = max(fails, key=lambda h: h['tension']) if fails else None
        if f is not None and f['legal_row'] is not None:
            p_ = next(j for j in range(7) if f['legal_row'][j] != f['nat'][j])
            verdict = (f['r'], p_, f['legal_row'][p_])
            why = "highest-tension row via best legal parse"
        else:
            verdict, why = (0, 5, 0), "degenerate fallback"

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

ledger = "R%02d:%d:%d" % verdict
_bad = sum(1 for r in range(N_ROWS)
           if check_digit(digits[8 * r:8 * r + 7].astype(int)) != digits[8 * r + 7]
           or not row_is_legal(r, digits[8 * r:8 * r + 8].astype(int)))
print("VERDICT %s  (%s)" % (ledger, why))
print("rows still not closing after repair: %d" % _bad)
print("total wall time %.0fs" % (time.time() - _T0))
# `restored`, `digits`, `ledger` are the submission quantities; the SUBMISSION
# cell below writes submission.npz from them.
