"""Joint structural decode: ward masking, bed pinning, close-margin ranking,
and corruption-parse verdict (the one-digit-wrong explanation)."""
import numpy as np

WARDS = [1, 2, 3, 4]
FORMULARY = {
    "05": (10, 40), "08": (5, 25), "11": (20, 60), "17": (15, 45),
    "23": (30, 90), "26": (10, 50), "34": (25, 75), "39": (5, 35),
    "42": (40, 95), "51": (20, 70), "63": (15, 55), "77": (35, 85),
}
WEIGHTS = np.array([7, 3, 1, 7, 3, 1, 7])

LEGAL_BODIES = []
for d0 in WARDS:
    for code, (lo, hi) in FORMULARY.items():
        c0, c1 = int(code[0]), int(code[1])
        for dose in range(lo, hi + 1):
            d5, d6 = dose // 10, dose % 10
            body = [d0, 0, 0, c0, c1, d5, d6]
            d7 = int(np.dot(WEIGHTS, body) % 10)
            LEGAL_BODIES.append([d0, c0, c1, d5, d6, d7])   # slots 0,3,4,5,6,7
LEGAL_BODIES = np.array(LEGAL_BODIES)                       # (K, 6)


def structural_lp(P):
    """Apply structural priors: pos0 in wards, pos1/2 pinned to the bed number."""
    P = np.asarray(P, dtype=np.float64).copy()
    for r in range(64):
        P[8 * r, :] = 0; P[8 * r, WARDS] = 1.0
        P[8 * r + 1, :] = 0; P[8 * r + 1, (r + 1) // 10] = 1.0
        P[8 * r + 2, :] = 0; P[8 * r + 2, (r + 1) % 10] = 1.0
    return np.log(np.clip(P, 1e-12, 1))


def row_decode(lp8):
    """lp8: (8, 10). Returns natural read, scores, margin, best closing row."""
    nat = [int(np.array(WARDS)[np.argmax(lp8[0, WARDS])])] \
        + [int(lp8[j].argmax()) for j in range(1, 8)]
    s_nat = float(sum(lp8[j, nat[j]] for j in range(8)))
    base = lp8[1, nat[1]] + lp8[2, nat[2]]
    best = (-np.inf, None)
    for rowk in LEGAL_BODIES:
        s = base + lp8[0, rowk[0]] + lp8[3, rowk[1]] + lp8[4, rowk[2]] \
            + lp8[5, rowk[3]] + lp8[6, rowk[4]] + lp8[7, rowk[5]]
        if s > best[0]:
            best = (float(s), rowk)
    s_close, rowk = best
    close_row = [rowk[0], nat[1], nat[2], rowk[1], rowk[2], rowk[3], rowk[4], rowk[5]]
    closes = (int(np.dot(WEIGHTS, nat[:7]) % 10) == nat[7])
    legal = _legal(nat)
    return dict(nat=nat, s_nat=s_nat, s_close=s_close, margin=s_nat - s_close,
                closes=closes, legal=legal, close_row=close_row)


def _legal(d):
    if d[0] not in WARDS:
        return False
    code = f"{d[3]}{d[4]}"
    if code not in FORMULARY:
        return False
    lo, hi = FORMULARY[code]
    return lo <= d[5] * 10 + d[6] <= hi


def corruption_parse(lp8, nat):
    """All one-digit-wrong explanations: written row = legal row with exactly one
    digit (pos 0..6, not bed) different. Returns [(score, pos, true_digit), ...] desc."""
    base = lp8[1, nat[1]] + lp8[2, nat[2]]
    out = []
    for rowk in LEGAL_BODIES:
        L = [rowk[0], nat[1], nat[2], rowk[1], rowk[2], rowk[3], rowk[4], rowk[5]]
        diff = [j for j in range(8) if L[j] != nat[j]]
        if len(diff) != 1:
            continue
        p = diff[0]
        if p in (1, 2, 7):
            continue
        s = base + sum(lp8[j, L[j]] for j in range(8) if j != p) + lp8[p, nat[p]]
        out.append((float(s), p, L[p]))
    out.sort(reverse=True)
    return out


def full_report(lp, top=5):
    dec = [row_decode(lp[8 * r:8 * r + 8]) for r in range(64)]
    dec_sorted = sorted(range(64), key=lambda r: -dec[r]['margin'])
    n_open = sum(1 for r in range(64) if not (dec[r]['legal'] and dec[r]['closes']))
    print("%d rows fail natural read" % n_open)
    print("top margin rows (most confidently NOT closing):")
    for r in dec_sorted[:top]:
        d = dec[r]
        cp = corruption_parse(lp[8 * r:8 * r + 8], d['nat'])
        best_cp = cp[0] if cp else None
        print("  R%02d read=%s legal=%s closes=%s margin=%.2f "
              "close_row=%s (dS=%.2f) corrupt%s"
              % (r, d['nat'], d['legal'], d['closes'], d['margin'],
                 d['close_row'], d['s_close'] - d['s_nat'],
                 (" -> pos%d should be %d (score gap %.2f)"
                  % (best_cp[1], best_cp[2], cp[1][0] - best_cp[0]) if best_cp and len(cp) > 1
                  else (" -> pos%d:%d" % (best_cp[1], best_cp[2])) if best_cp else "none")))
    return dec
