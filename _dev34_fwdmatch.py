import time
import numpy as np
import torch
import _utils as u
from _utils import warp_batch, hblur_batch, make_features_batch, nrmse, corrupt_batch
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


def both(Zz):
    FA, sig = make_features_batch(Zz)
    FC, _ = make_features_C_batch(Zz)
    with torch.no_grad():
        RA = da(torch.from_numpy(FA)).numpy()[:, 0]
        RC = dc(torch.from_numpy(FC)).numpy()[:, 0]
    return np.clip(0.4 * RA + 0.6 * RC, 0, 1), sig, FA[:, 0]


rng = np.random.default_rng(999)
NSIM = 500
sidx = rng.integers(0, len(XALL), NSIM)
ZS, X0S, SIGS = corrupt_batch(XALL[sidx], rng, t_mix=MIX)

QZ = np.concatenate([ZS, CN, Z])          # sim first, then calib, then real
N1, N2 = NSIM, NSIM + 24
RQ, SIGQ, ZDQ = both(QZ)
print('loaded+regression %.0fs' % (time.time() - T0), flush=True)


def pool14(A):
    return A.reshape(len(A), 14, 2, 14, 2).mean((2, 4))


def norm2(A):
    A = A.reshape(len(A), -1).astype(np.float32)
    A = A - A.mean(1, keepdims=True)
    return A / (np.sqrt((A ** 2).sum(1, keepdims=True)) + 1e-9)


# reuse coarse candidates from dev30 (rows: 0..511 = real Z, 512..535 = calib),
# remapped to the new query order [sim, calib, realZ]; sim rows rely on the
# recall-boost pass below
TOPK = 12
_co = np.load('_coarse.npz')
cand_idx = np.zeros((len(QZ), TOPK), dtype=np.int64)
cand_var = np.zeros((len(QZ), TOPK, 2))
cand_sc0 = np.full((len(QZ), TOPK), -2.0)
n_prev = min(_co['idx'].shape[1], TOPK)
remap = None  # placeholder (mapping done via src_rows below)
src_rows = np.concatenate([np.full(NSIM, -1, dtype=int),
                           np.arange(512, 512 + 24), np.arange(0, 512)])
for i, o in enumerate(src_rows):
    if o >= 0:
        cand_idx[i, :n_prev] = _co['idx'][o, :n_prev]
        cand_sc0[i, :n_prev] = _co['sc'][o, :n_prev]
        cand_var[i, :n_prev] = _co['var'][o, :n_prev]
# boost recall with a no-rotation NCC pass on raw+blurred 14x14
XALL14 = pool14(XALL)
Traw = norm2(XALL14)
Tblu = norm2(hblur_batch(XALL14, np.full(len(XALL14), 3)))
ZD14 = norm2(pool14(np.clip(ZDQ, 0, 1.35)))
ZD14b = norm2(pool14(hblur_batch(np.clip(ZDQ, 0, 1.35), np.full(len(QZ), 3))))
for Tb, Qb in ((Traw, ZD14), (Tblu, ZD14b)):
    S = (Tb @ Qb.T).T.astype(np.float64)
    for i in range(len(QZ)):
        top = np.argpartition(S[i], -12)[-12:]
        seen = {int(cand_idx[i, k]) for k in range(TOPK) if cand_sc0[i, k] > -2}
        news = sorted(((float(S[i, j]), int(j)) for j in top if int(j) not in seen),
                      reverse=True)
        for s_, j in news:
            worst = int(np.argmin(cand_sc0[i]))
            if s_ > cand_sc0[i, worst]:
                cand_sc0[i, worst] = s_; cand_idx[i, worst] = j
                cand_var[i, worst] = (0.0, 1.0)
print('coarse candidates loaded/expanded %.0fs' % (time.time() - T0), flush=True)

LS = (1, 3, 7)
DTH = np.array([-2., 0., 2.])
DSC = np.array([-0.04, 0., 0.04])
SH5 = np.array([-2., -1., 0., 1., 2.])
FDX, FDY = np.meshgrid(SH5, SH5, indexing='ij')
FDX, FDY = FDX.ravel(), FDY.ravel()
NSH = 25
G_th, G_sc, G_dx, G_dy = np.meshgrid(
    np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-0.03, -0.015, 0., 0.015, 0.03]),
    np.array([-1., -0.5, 0., 0.5, 1.]), np.array([-1., -0.5, 0., 0.5, 1.]), indexing='ij')
G_th, G_sc, G_dx, G_dy = (g.ravel() for g in (G_th, G_sc, G_dx, G_dy))


def fit_score(Wb, z):
    """least-scores z ~= g*Wb + b over each warped image; returns -residual variance."""
    Wf = Wb.reshape(len(Wb), -1)
    zf = z.reshape(-1)
    zm = zf - zf.mean()
    Wm = Wf - Wf.mean(1, keepdims=True)
    vw = (Wm ** 2).sum(1)
    cov = Wm @ zm
    g = np.where(vw > 1e-8, cov / np.maximum(vw, 1e-8), 0.0)
    b = zf.mean() - g * Wf.mean(1)
    pred = g[:, None] * Wf + b[:, None]
    resid = ((pred - zf[None]) ** 2).mean(1)
    score = 1.0 - resid / (zm @ zm * 2 / len(zf) + 1e-12) / 0.5
    return -resid, g, b


tmpl_out = np.zeros_like(RQ)
qual = np.zeros(len(QZ))
best_src = np.zeros(len(QZ), dtype=np.int64)
for i in range(len(QZ)):
    cidx = cand_idx[i]
    thc = cand_var[i, :, 0]; scc = cand_var[i, :, 1]
    z = ZDQ[i]
    zf = z.reshape(-1)
    # stage 1: per-candidate shift scan at coarse warp
    th1 = np.repeat(thc, NSH); sc1 = np.clip(np.repeat(scc, NSH), 0.85, 1.15)
    dx1 = np.tile(FDX, TOPK); dy1 = np.tile(FDY, TOPK)
    W1 = np.clip(warp_batch(np.repeat(XALL[cidx], NSH, 0), th1, sc1, dx1, dy1), 0, 1)
    s1all = np.empty((len(LS), len(W1)))
    for li, L in enumerate(LS):
        s1all[li] = fit_score(hblur_batch(W1, np.full(len(W1), L)), z)[0]
    s1 = s1all.max(0)
    s1m = s1.reshape(TOPK, NSH)
    top3 = np.argsort(-s1m.max(1))[:3]
    # stage 2: theta/scale/shift refine for top-3
    rows = []
    for c in top3:
        for dth in DTH:
            for dsc in DSC:
                rows.append((int(c), thc[c] + dth, float(np.clip(scc[c] + dsc, 0.85, 1.15))))
    base = np.array(rows)
    nbase = len(base)
    th2 = np.repeat(base[:, 1], NSH); sc2 = np.repeat(base[:, 2], NSH)
    dx2 = np.tile(FDX, nbase); dy2 = np.tile(FDY, nbase)
    W2 = np.clip(warp_batch(np.repeat(XALL[cidx[base[:, 0].astype(int)]], NSH, 0),
                            th2, sc2, dx2, dy2), 0, 1)
    s2all = np.empty((len(LS), len(W2)))
    for li, L in enumerate(LS):
        s2all[li] = fit_score(hblur_batch(W2, np.full(len(W2), L)), z)[0]
    s2 = s2all.max(0)
    j2 = int(np.argmax(s2))
    li2 = int(s2all[:, j2].argmax())
    jb = j2 // NSH
    cw = int(base[jb, 0]); th_w = base[jb, 1]; sc_w = base[jb, 2]
    dx_w = FDX[j2 % NSH]; dy_w = FDY[j2 % NSH]
    # stage 3: super-fine
    th3 = th_w + G_th; sc3 = np.clip(sc_w + G_sc, 0.85, 1.15)
    dx3 = dx_w + G_dx; dy3 = dy_w + G_dy
    W3 = np.clip(warp_batch(np.repeat(XALL[cidx[cw]:cidx[cw] + 1], len(th3), 0),
                            th3, sc3, dx3, dy3), 0, 1)
    s3all = np.empty((len(LS), len(W3)))
    for li, L in enumerate(LS):
        s3all[li] = fit_score(hblur_batch(W3, np.full(len(W3), L)), z)[0]
    s3 = s3all.max(0)
    k = int(np.argmax(s3))
    # quality: explained variance ratio at best fit
    Wbf = hblur_batch(W3[k:k + 1], np.full(1, LS[int(s3all[:, k].argmax())]))[0]
    res, g_, b_ = fit_score(Wbf[None], z)
    varz = ((zf - zf.mean()) ** 2).mean()
    qual[i] = float(1.0 + res[0] / (varz + 1e-12))
    tmpl_out[i] = W3[k]
    best_src[i] = cidx[cw]
    if i == NSIM - 1:
        hit = (YALL[best_src[:NSIM]] == YALL[sidx]).mean()
        print('SIM identity hit after %d crops: %.3f  qual med %.3f (%.0fs)'
              % (NSIM, hit, np.median(qual[:NSIM]), time.time() - T0), flush=True)
    if i % 128 == 0 or i in (NSIM, N2 - 1):
        print('fine %d/%d %.0fs q=%.3f' % (i, len(QZ), time.time() - T0, qual[i]), flush=True)

np.savez_compressed('_tmpl2.npz', tmpl=tmpl_out, qual=qual, R=RQ, SIG=SIGQ, src=best_src)


def rep(tag, R_, X0):
    e = np.sqrt(((R_ - X0) ** 2).mean((1, 2))) / (np.sqrt((X0 ** 2).mean((1, 2))) + 1e-8)
    print('%s pooled %.4f per-crop %.4f' % (tag, nrmse(R_, X0), e.mean()))


TS, RS, QS = tmpl_out[:NSIM], RQ[:NSIM], qual[:NSIM]
rep('SIM ens     ', RS, X0S)
rep('SIM template', TS, X0S)
print('sim identity hit %.3f' % (YALL[best_src[:NSIM]] == YALL[sidx]).mean())
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    G = RS.copy(); g = QS >= th
    G[g] = TS[g]
    rep('SIM gate %.2f (%d tmpl)' % (th, int(g.sum())), G, X0S)
for q0, q1 in ((0.35, 0.55), (0.4, 0.6), (0.45, 0.65)):
    w = np.clip((QS - q0) / (q1 - q0), 0, 1)[:, None, None]
    rep('SIM blend %.2f-%.2f' % (q0, q1), w * TS + (1 - w) * RS, X0S)
TC, RC, QC = tmpl_out[NSIM:N2], RQ[NSIM:N2], qual[NSIM:N2]
rep('CAL ens     ', RC, CC)
rep('CAL template', TC, CC)
for th in (0.4, 0.5, 0.6):
    G = RC.copy(); g = QC >= th
    G[g] = TC[g]
    rep('CAL gate %.2f' % th, G, CC)
print('total %.0fs' % (time.time() - T0), flush=True)
