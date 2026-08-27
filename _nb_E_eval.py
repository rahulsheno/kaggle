# ==================== READ THE LEDGER ====================
# 1. restore all 512 crops with the ensemble
# 2. two independent reads (restored view + raw view), shift-averaged over small
#    translations, sharpened to a common temperature and blended; the temperature and
#    blend weight are picked on the 128 bed-anchor digits
# 3. verdict: for every row, compare the evidence that its writing is one of the legal
#    closings against the evidence that it is a legal closing with exactly one digit
#    mis-written, marginalising over all legal parses. The row with the strongest
#    one-digit-corruption explanation is the failing row, and the repair is the verdict.
T0 = time.time()
restored, SIGZ = restore_batch(Z)
print("restoration done in %.0fs" % (time.time() - T0))
print("restored range [%.3f, %.3f]  mean %.4f"
      % (restored.min(), restored.max(), restored.mean()))
probe(lambda z: restore_batch(np.asarray(z)[None])[0][0])

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
        F, _ = make_features_C(Zs)
        with torch.no_grad():
            outs.append(torch.softmax(CLS_N(torch.from_numpy(F)), 1).numpy())
    return np.mean(outs, 0)


Pm = read_restored(restored, SIGZ)
Pn = read_raw(Z)
_acc = lambda Qk: float((Qk[ANCH_IDX].argmax(1) == ANCH_LAB).mean())
print("anchor accuracy: restored-read %.3f | raw-read %.3f" % (_acc(Pm), _acc(Pn)))

best_T, best_w, best_acc = 1.0, 0.5, -1.0
for T in np.arange(0.5, 2.01, 0.05):
    Qm = Pm ** (1 / T); Qm /= Qm.sum(1, keepdims=True)
    Qn = Pn ** (1 / T); Qn /= Qn.sum(1, keepdims=True)
    for w in np.arange(0.0, 1.01, 0.05):
        a = _acc(w * Qm + (1 - w) * Qn)
        if a > best_acc:
            best_acc, best_T, best_w = a, T, w
TEMP, W_BLEND = best_T, best_w
Qm = Pm ** (1 / TEMP); Qm /= Qm.sum(1, keepdims=True)
Qn = Pn ** (1 / TEMP); Qn /= Qn.sum(1, keepdims=True)
P = W_BLEND * Qm + (1 - W_BLEND) * Qn
print("blend: T=%.2f w=%.2f -> anchor acc %.4f" % (TEMP, W_BLEND, _acc(P)))

# ---- verdict by marginalised evidence ratios -------------------------------
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
LB = np.array(LB)                          # (K,6): values for slots 0,3,4,5,6,7
SLOTS = [0, 3, 4, 5, 6, 7]
CORRUPTIBLE = [0, 3, 4, 5, 6]


def _lse(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def verdict_scores(lp_all):
    """log Z1_r(p,v) - log Z0_r for every row: the evidence that the writing is a
    legal closing with digit p mis-written as v, against the evidence that it is a
    legal closing, both marginalised over every legal parse."""
    out = []
    for r in range(N_ROWS):
        l8 = lp_all[8 * r:8 * r + 8]
        bed = ((r + 1) // 10, (r + 1) % 10)
        base = l8[1, bed[0]] + l8[2, bed[1]]
        lslot = np.stack([l8[s, LB[:, k]] for k, s in enumerate(SLOTS)], 1)
        sfull = base + lslot.sum(1)
        lz0 = _lse(sfull)
        sother = base + lslot.sum(1, keepdims=True) - lslot        # (K,6)
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
i_best = int(np.argmax(vs_flat))
r_v, pv = divmod(i_best, 70)
pos_v, val_v = divmod(pv, 10)
n_near = int((vs_flat.max(1) > vs_flat.max() - 1.0).sum())

# the ledger string must carry what the digit SHOULD have been: the digit at pos_v in
# the most likely legal closing of the verdict row given everything written there
l8v = lg[8 * r_v:8 * r_v + 8]
bed = ((r_v + 1) // 10, (r_v + 1) % 10)
lslot = np.stack([l8v[s, LB[:, k]] for k, s in enumerate(SLOTS)], 1)
base = l8v[1, bed[0]] + l8v[2, bed[1]]
s_o = base + lslot.sum(1) - lslot[:, CORRUPTIBLE.index(pos_v)]
i_L = int(np.argmax(s_o))
true_v = int(LB[i_L, CORRUPTIBLE.index(pos_v)])
verdict = (r_v, pos_v, true_v)
why = ("marginalised corruption evidence: row %d, digit %d written as %d, should be %d "
       "(%d rows within 1 nat of the best)" % (r_v, pos_v, val_v, true_v, n_near))

# ---- digits + structural cleanup -------------------------------------------


def best_legal_parse(r, lp):
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


digits = P.argmax(1).copy()
fails, nat_of = [], {}
for r in range(N_ROWS):
    p8, lp8 = P[8 * r:8 * r + 8], lg[8 * r:8 * r + 8]
    bed = r + 1
    nat = [int(np.array(WARDS)[np.argmax(p8[0, WARDS])]), bed // 10, bed % 10] \
        + [int(p8[j].argmax()) for j in range(3, 8)]
    nat_of[r] = nat
    if not (row_is_legal(r, nat) and check_digit(nat[:7]) == nat[7]):
        sc_legal, legal_row = best_legal_parse(r, lp8)
        fails.append(dict(r=r, nat=nat,
                          logconf=float(sum(lp8[j, nat[j]] for j in range(8))),
                          legal_row=legal_row))
print("%d of %d rows fail the natural read:" % (len(fails), N_ROWS))
for f in fails:
    print("  row %2d read=%s legal=%s closes=%s logconf=%.2f"
          % (f['r'], f['nat'], row_is_legal(f['r'], f['nat']),
             check_digit(f['nat'][:7]) == f['nat'][7], f['logconf']))

# final digits: natural read everywhere; structurally pinned positions overwritten;
# failed rows replaced by their best legal parse unless they are the verdict row -
# `digits` says what is written on the chart, `ledger` says what it should be.
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
           if check_digit(digits[8 * r:8 * r + 7].astype(int).tolist()) != digits[8 * r + 7]
           or not row_is_legal(r, digits[8 * r:8 * r + 8].astype(int).tolist()))
print("VERDICT %s   (%s)" % (ledger, why))
print("rows still not closing after repair: %d (the verdict row is expected to be one)"
      % _bad)
print("digits histogram:", np.bincount(digits, minlength=10).tolist())
print("total wall time %.0fs" % (time.time() - T0))
