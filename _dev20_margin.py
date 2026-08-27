import numpy as np

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
LB = np.array(LB)                     # (K,6): slots 0,3,4,5,6,7
SLOTS = [0, 3, 4, 5, 6, 7]


def margins(P):
    lp = np.log(np.clip(np.asarray(P), 1e-12, 1))
    out = []
    for r in range(64):
        l8 = lp[8 * r:8 * r + 8]
        bed = ((r + 1) // 10, (r + 1) % 10)
        base = l8[1, bed[0]] + l8[2, bed[1]]
        sfull = base + l8[SLOTS[0], LB[:, 0]] + l8[SLOTS[1], LB[:, 1]] \
            + l8[SLOTS[2], LB[:, 2]] + l8[SLOTS[3], LB[:, 3]] \
            + l8[SLOTS[4], LB[:, 4]] + l8[SLOTS[5], LB[:, 5]]
        sH0 = sfull.max()
        sH1 = -1e9; cp = None
        for k_, p in enumerate(SLOTS[:5]):       # corruptible slots 0,3,4,5,6
            cand = sfull[:, None] + l8[p, :][None, :] - l8[p, LB[:, k_]][:, None]
            i, v = np.unravel_index(np.argmax(cand), cand.shape)
            if v == LB[i, k_]:
                cand[i, v] = -np.inf
                i, v = np.unravel_index(np.argmax(cand), cand.shape)
            if cand[i, v] > sH1:
                sH1 = cand[i, v]; cp = (p, int(v), int(i))
        out.append((r, float(sH1 - sH0), cp))
    return out


if __name__ == '__main__':
    pr = np.load('_probs_all.npz')
    for k in ('Pm', 'Pn', 'Pmft', 'Pnft'):
        ms = margins(pr[k])
        top = sorted(ms, key=lambda t: -t[1])[:8]
        print(k, [(r, round(m, 2), (c[0], c[1])) for r, m, c in top])
