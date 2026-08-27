import numpy as np
from scipy.special import logsumexp

WARDS = [1, 2, 3, 4]
FORMULARY = {'05': (10, 40), '08': (5, 25), '11': (20, 60), '17': (15, 45),
             '23': (30, 90), '26': (10, 50), '34': (25, 75), '39': (5, 35),
             '42': (40, 95), '51': (20, 70), '63': (15, 55), '77': (35, 85)}
WEIGHTS = np.array([7, 3, 1, 7, 3, 1, 7])

LB = []
for d0 in WARDS:
    for code, (lo, hi) in FORMULARY.items():
        c0, c1 = int(code[0]), int(code[1])
        for dose in range(lo, hi + 1):
            d5, d6 = dose // 10, dose % 10
            d7v = int((WEIGHTS[0] * d0 + WEIGHTS[3] * c0 + WEIGHTS[4] * c1
                       + WEIGHTS[5] * d5 + WEIGHTS[6] * d6) % 10)
            LB.append((d0, c0, c1, d5, d6, d7v))
LB = np.array(LB)                     # (K,6): values for slots 0,3,4,5,6,7
SLOTS = [0, 3, 4, 5, 6, 7]


def verdict_scores(P):
    """For each row r and corruption (p, v): log Z1_r(p,v) - log Z0_r,
    where Z0 marginalises over all legal closings and Z1 over all legal
    closings with one written digit changed."""
    lp = np.log(np.clip(np.asarray(P), 1e-12, 1))
    K = len(LB)
    out = []
    for r in range(64):
        l8 = lp[8 * r:8 * r + 8]
        bed = ((r + 1) // 10, (r + 1) % 10)
        base = l8[1, bed[0]] + l8[2, bed[1]]
        lslot = np.stack([l8[s, LB[:, k]] for k, s in enumerate(SLOTS)], 1)  # (K,6)
        sfull = base + lslot.sum(1)                                           # (K,)
        lz0 = logsumexp(sfull)
        best = (-1e9, None, None)
        for k_, p in enumerate(SLOTS[:5]):           # corruptible: 0,3,4,5,6
            s_others = base + lslot.sum(1) - lslot[:, k_]                     # (K,)
            for v in range(10):
                mask = LB[:, k_] != v                # written digit must differ
                if not mask.any():
                    continue
                a_ = np.where(mask, s_others + l8[p, v], -np.inf)
                lz1 = logsumexp(a_)
                s = lz1 - lz0
                if s > best[0]:
                    best = (s, p, v)
        out.append((r, best[0], best[1], best[2]))
    return out


if __name__ == '__main__':
    pr = np.load('_probs_all.npz')
    ANCH_IDX = np.array([8 * r + j for r in range(64) for j in (1, 2)])
    ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10
                         for r in range(64) for j in (1, 2)])
    best_blend = None
    for k1 in pr.files:
        for k2 in pr.files:
            P1, P2 = pr[k1], pr[k2]
            for T in (0.5, 0.7, 1.0, 1.3):
                Q1 = P1 ** (1 / T); Q1 /= Q1.sum(1, keepdims=True)
                Q2 = P2 ** (1 / T); Q2 /= Q2.sum(1, keepdims=True)
                for w in (0.3, 0.5, 0.7):
                    P = w * Q1 + (1 - w) * Q2
                    a = (P[ANCH_IDX].argmax(1) == ANCH_LAB).mean()
                    if best_blend is None or a > best_blend[0]:
                        best_blend = (a, k1, k2, T, w, P)
    a, k1, k2, T, w, P = best_blend
    print('best blend %s+%s T=%.1f w=%.1f anchor acc %.4f' % (k1, k2, T, w, a))
    vs = verdict_scores(P)
    top = sorted(vs, key=lambda t: -t[1])[:12]
    print('bayes-verdict top:', [(r, round(s, 3), (p, v)) for r, s, p, v in top])
