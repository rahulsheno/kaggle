import time
import numpy as np
import torch
import _utils as u
from _utils import warp_batch, make_features_batch, nrmse, corrupt_batch
from _dev15_cls_final import DenNet
from _dev28_den_v3 import make_features_C_batch

T0 = time.time()
torch.set_num_threads(8)
MIX = (0.80, 25.0, 110.0, 0.0, 200.0)

X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
Y_TRAIN = np.load('y_kannada_MNIST_train.npz')['arr_0'].astype(np.int64)
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
Y_DIG = np.load('y_dig_MNIST.npz')['arr_0'].astype(np.int64)
XALL = np.concatenate([X_TRAIN, X_DIG])
YALL = np.concatenate([Y_TRAIN, Y_DIG])
_a = np.load('arogya_archive_v1.npz')
Z = _a['Z'].astype(np.float64)
CN = _a['calib_noisy'].astype(np.float64)
CC = _a['calib_clean'].astype(np.float64)

da = DenNet(2); da.load_state_dict(torch.load('_den_a.pt')); da.eval()
dc = DenNet(4); dc.load_state_dict(torch.load('_den_c.pt')); dc.eval()


def ens(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = da(torch.from_numpy(FA)).numpy()[:, 0]
        RC = dc(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip((RA + RC) / 2, 0, 1), sig


rng = np.random.default_rng(4242)
NSIM = 600
sidx = rng.integers(0, len(XALL), NSIM)
ZS, X0S, SIGS = corrupt_batch(XALL[sidx], rng, t_mix=MIX)

QUERIES_Z = np.concatenate([Z, CN, ZS])
NREAL, NCAL = len(Z), len(CN)
print('loaded %.0fs' % (time.time() - T0), flush=True)

RQ, SIGQ = ens(QUERIES_Z)
print('regression done %.0fs  calib NRMSE %.4f'
      % (time.time() - T0, nrmse(RQ[NREAL:NREAL + NCAL], CC)), flush=True)


def pool14(A):
    return A.reshape(len(A), 14, 2, 14, 2).mean((2, 4))


VARIANTS = [(th, sc) for th in (-12, -6, 0, 6, 12) for sc in (0.90, 1.00, 1.12)]
Q14 = pool14(np.clip(RQ, 0, 1)).astype(np.float32).reshape(len(RQ), -1)
Q14 = Q14 - Q14.mean(1, keepdims=True)
Q14 /= (np.sqrt((Q14 ** 2).sum(1, keepdims=True)) + 1e-9)

XALL14 = pool14(XALL)
TOPK = 8
cand_idx = np.zeros((len(QUERIES_Z), TOPK), dtype=np.int64)
cand_var = np.zeros((len(QUERIES_Z), TOPK, 2))
cand_sc0 = np.full((len(QUERIES_Z), TOPK), -2.0)
CH = 24000
for v_i, (th, sc) in enumerate(VARIANTS):
    T14 = []
    for c0 in range(0, len(XALL14), CH):
        blk = XALL14[c0:c0 + CH]
        W = np.clip(warp_batch(blk, np.full(len(blk), th), np.full(len(blk), sc),
                               np.zeros(len(blk)), np.zeros(len(blk))), 0, 1)
        T14.append(W)
    T14 = np.concatenate(T14).astype(np.float32).reshape(len(XALL14), -1)
    T14 = T14 - T14.mean(1, keepdims=True)
    T14 /= (np.sqrt((T14 ** 2).sum(1, keepdims=True)) + 1e-9)
    S = (T14 @ Q14.T).T                                # (Q, 70240)
    S64 = S.astype(np.float64)
    for i in range(len(QUERIES_Z)):
        top = np.argpartition(S64[i], -24)[-24:]
        m_i, m_s, m_v = [], [], []
        seen = {}
        for k in range(TOPK):
            if cand_sc0[i, k] > -2:
                seen.setdefault(int(cand_idx[i, k]), k)
        for o in top:
            j = int(o)
            if j in seen:
                k = seen[j]
                if S64[i, j] > cand_sc0[i, k]:
                    cand_sc0[i, k] = S64[i, j]; cand_var[i, k] = (th, sc)
            else:
                m_i.append(j); m_s.append(float(S64[i, j]))
        order = np.argsort(-np.array(m_s)) if m_s else []
        for o in order:
            worst = int(np.argmin(cand_sc0[i]))
            if m_s[o] > cand_sc0[i, worst]:
                cand_sc0[i, worst] = m_s[o]
                cand_idx[i, worst] = m_i[o]
                cand_var[i, worst] = (th, sc)
    print('variant %d/%d done %.0fs' % (v_i + 1, len(VARIANTS), time.time() - T0), flush=True)

np.savez_compressed('_coarse.npz', idx=cand_idx, sc=cand_sc0, var=cand_var)
print('coarse matching saved', flush=True)

DTH2 = np.array([-2., -1., 0., 1., 2.])
DSC2 = np.array([-0.03, 0., 0.03])
SH5 = np.array([-2., -1., 0., 1., 2.])
FDX, FDY = np.meshgrid(SH5, SH5, indexing='ij')
FDX, FDY = FDX.ravel(), FDY.ravel()
G_th, G_sc, G_dx, G_dy = np.meshgrid(
    np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-0.03, -0.015, 0., 0.015, 0.03]),
    np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-1., -0.5, 0., 0.5, 1.]), indexing='ij')
G_th, G_sc, G_dx, G_dy = (g.ravel() for g in (G_th, G_sc, G_dx, G_dy))

tmpl_out = np.zeros_like(RQ)
qual = np.zeros(len(QUERIES_Z))
best_src = np.zeros(len(QUERIES_Z), dtype=np.int64)
match_lab = np.zeros(len(QUERIES_Z), dtype=np.int64)


def _norm_rows(A):
    A = A - A.mean((1, 2), keepdims=True)
    n = np.sqrt((A ** 2).sum((1, 2), keepdims=True)) + 1e-9
    return (A / n).reshape(len(A), -1)


dx_block = np.tile(FDX, 1)
dy_block = np.tile(FDY, 1)
NSH = len(SH5) ** 2
for i in range(len(QUERIES_Z)):
    cidx = cand_idx[i]
    thc = cand_var[i, :, 0]; scc = cand_var[i, :, 1]
    r = _norm_rows(np.clip(RQ[i:i + 1], 0, 1))[0]
    # ---- stage 1: shifts only at each candidate's coarse warp (TOPK*25) ----
    th1 = np.repeat(thc, NSH); sc1 = np.clip(np.repeat(scc, NSH), 0.85, 1.15)
    dx1 = np.tile(FDX, TOPK); dy1 = np.tile(FDY, TOPK)
    imgs1 = np.repeat(XALL[cidx], NSH, 0)
    W1 = np.clip(warp_batch(imgs1, th1, sc1, dx1, dy1), 0, 1)
    s1 = _norm_rows(W1) @ r
    s1m = s1.reshape(TOPK, NSH)
    best_shift_each = s1m.argmax(1)
    top3 = np.argsort(-s1m.max(1))[:3]
    # ---- stage 2: refine theta/scale/shift for top-3 (3*5*3*25) ----
    rows = []
    for c in top3:
        for dth in DTH2:
            for dsc in DSC2:
                rows.append((c, thc[c] + dth,
                             float(np.clip(scc[c] + dsc, 0.85, 1.15))))
    base = np.array(rows)                       # (3*5*3, 3)
    nbase = len(base)
    th2 = np.repeat(base[:, 1], NSH); sc2 = np.repeat(base[:, 2], NSH)
    dx2 = np.tile(FDX, nbase); dy2 = np.tile(FDY, nbase)
    imgs2 = np.repeat(XALL[cidx[base[:, 0].astype(int)]], NSH, 0)
    W2 = np.clip(warp_batch(imgs2, th2, sc2, dx2, dy2), 0, 1)
    s2 = (_norm_rows(W2) @ r).reshape(nbase, NSH)
    jb, js = np.unravel_index(int(np.argmax(s2)), s2.shape)
    cw = int(base[jb, 0]); th_w = base[jb, 1]; sc_w = base[jb, 2]
    dx_w = FDX[js]; dy_w = FDY[js]
    # ---- stage 3: super-fine around winner (625) ----
    th3 = th_w + G_th; sc3 = np.clip(sc_w + G_sc, 0.85, 1.15)
    dx3 = dx_w + G_dx; dy3 = dy_w + G_dy
    W3 = np.clip(warp_batch(np.repeat(XALL[cidx[cw]:cidx[cw] + 1], len(th3), 0),
                            th3, sc3, dx3, dy3), 0, 1)
    s3 = _norm_rows(W3) @ r
    k = int(np.argmax(s3))
    qual[i] = float(s3[k])
    tmpl_out[i] = W3[k]
    best_src[i] = cidx[cw]
    match_lab[i] = YALL[cidx[cw]]
    if i % 128 == 0:
        print('fine %d/%d %.0fs (q=%.3f)' % (i, len(QUERIES_Z), time.time() - T0, qual[i]),
              flush=True)

np.savez_compressed('_tmpl_rest.npz', tmpl=tmpl_out, qual=qual, R=RQ, SIG=SIGQ,
                    lab=match_lab, src=best_src)


def report(tag, R_, X0):
    e = np.sqrt(((R_ - X0) ** 2).mean((1, 2))) / (np.sqrt((X0 ** 2).mean((1, 2))) + 1e-8)
    print('%s pooled %.4f per-crop %.4f p90 %.3f' % (tag, nrmse(R_, X0), e.mean(),
                                                      np.percentile(e, 90)))


i0, i1 = NREAL, NREAL + NCAL
i2, i3 = i1, i1 + NSIM
TS = tmpl_out[i2:i3]; RS = RQ[i2:i3]
report('SIM ens     ', RS, X0S)
report('SIM template', TS, X0S)
hit = (match_lab[i2:i3] == YALL[sidx])
print('sim source-label hit %.3f  qual med %.3f min %.3f'
      % (hit.mean(), np.median(qual[i2:i3]), qual[i2:i3].min()))
for th in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
    G = RS.copy(); g = qual[i2:i3] >= th
    G[g] = TS[g]
    report('SIM gate %.2f  (%d tmpl)' % (th, g.sum()), G, X0S)
for q0, q1 in ((0.45, 0.65), (0.5, 0.7), (0.55, 0.75)):
    w = np.clip((qual[i2:i3] - q0) / (q1 - q0), 0, 1)[:, None, None]
    report('SIM blend %.2f-%.2f' % (q0, q1), w * TS + (1 - w) * RS, X0S)
report('CAL ens     ', RQ[i0:i1], CC)
report('CAL template', tmpl_out[i0:i1], CC)
for th in (0.5, 0.6, 0.7):
    G = RQ[i0:i1].copy(); g = qual[i0:i1] >= th
    G[g] = tmpl_out[i0:i1][g]
    report('CAL gate %.2f' % th, G, CC)
for q0, q1 in ((0.45, 0.65), (0.5, 0.7)):
    w = np.clip((qual[i0:i1] - q0) / (q1 - q0), 0, 1)[:, None, None]
    report('CAL blend %.2f-%.2f' % (q0, q1),
           w * tmpl_out[i0:i1] + (1 - w) * RQ[i0:i1], CC)
print('total %.0fs' % (time.time() - T0), flush=True)
