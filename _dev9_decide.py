import numpy as np

WEIGHTS = np.array([7, 3, 1, 7, 3, 1, 7])
WARDS = [1, 2, 3, 4]
FORMULARY = {
    "05": (10, 40), "08": (5, 25), "11": (20, 60), "17": (15, 45),
    "23": (30, 90), "26": (10, 50), "34": (25, 75), "39": (5, 35),
    "42": (40, 95), "51": (20, 70), "63": (15, 55), "77": (35, 85),
}
INV = {1: 1, 3: 7, 7: 3, 9: 9}

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

def solve_repair(nat, pos, rhs):
    body = nat[:7].copy()
    s = int(sum(WEIGHTS[j] * body[j] for j in range(7) if j != pos))
    return ((rhs - s) % 10) * INV[WEIGHTS[pos]] % 10

def main_repairs(r, nat):
    d7 = nat[7]
    reps = []
    for pos in (0, 3, 4, 5, 6):
        v = solve_repair(nat, pos, d7)
        if v == nat[pos]:
            continue
        body = nat[:7].copy(); body[pos] = v
        if pos == 0 and v not in WARDS:
            continue
        if row_is_legal(r, body + [d7]):
            reps.append((pos, v))
    return reps

def gen_repairs(r, nat, lp, pmin=-4.0):
    reps = []
    for pos in (0, 3, 4, 5, 6):
        for d7v in range(10):
            if lp[7, d7v] < pmin:
                continue
            v = solve_repair(nat, pos, d7v)
            if v == nat[pos]:
                continue
            body = nat[:7].copy(); body[pos] = v
            if pos == 0 and v not in WARDS:
                continue
            if row_is_legal(r, body + [d7v]):
                score = sum(lp[j, nat[j]] for j in range(7) if j != pos) + lp[7, d7v]
                reps.append((pos, v, d7v, float(score)))
    return reps

def natural_row(r, p8):
    bed = r + 1
    d0 = int(np.array([1, 2, 3, 4])[np.argmax(p8[0, [1, 2, 3, 4]])])
    return [d0, bed // 10, bed % 10] + [int(p8[j].argmax()) for j in range(3, 8)]

def adjudicate(P, verbose=True):
    lg = np.log(np.clip(P, 1e-12, 1))
    D = P.argmax(1)
    fails = []
    nat_of = {}
    for r in range(64):
        p8 = P[8 * r: 8 * r + 8]
        lp8 = lg[8 * r: 8 * r + 8]
        nat = natural_row(r, p8)
        legal = row_is_legal(r, nat)
        closes = check_digit(nat[:7]) == nat[7]
        nat_of[r] = nat
        if not (legal and closes):
            mr = main_repairs(r, nat)
            conf = float(sum(lp8[j, nat[j]] for j in range(8)))
            score_legal, legal_row = best_legal_parse(r, lp8)
            score_nat = float(sum(lp8[j, nat[j]] for j in range(8)))
            gr = [] if mr else gen_repairs(r, nat, lp8)
            fails.append(dict(r=r, nat=nat, legal=legal, closes=closes, mr=mr, gr=gr,
                              logconf=conf, tension=score_legal - score_nat,
                              legal_row=legal_row))
    if verbose:
        print(f"--- {len(fails)} rows fail the natural read ---")
        for f in fails:
            tag = f"mr={[(p, v) for p, v in f['mr']]}" if f['mr'] else f"gr(best)={max(f['gr'], key=lambda h: h[3]) if f['gr'] else None}"
            print(f"  row {f['r']:2d} read={f['nat']} legal={f['legal']} closes={f['closes']} "
                  f"{tag} logconf={f['logconf']:.2f} tension={f['tension']:+.2f}")
    # tier 1: exactly one main repair
    t1 = [f for f in fails if len(f['mr']) == 1]
    verdict = None; why = ''
    if t1:
        t1f = max(t1, key=lambda f: f['logconf'])
        p_, v_ = t1f['mr'][0]
        verdict = (t1f['r'], p_, v_)
        why = f"unique main repair; best logconf {t1f['logconf']:.2f} among {len(t1)} single-repair rows"
    else:
        t2 = [f for f in fails if f['mr']]
        pool = t2 if t2 else [f for f in fails if f['gr']]
        if pool:
            if t2:
                f = max(pool, key=lambda f: max(f['logconf'], -1e9))
                # pick the repair consistent with highest-posterior written digit
                p_, v_ = f['mr'][0]
                verdict = (f['r'], p_, v_)
                why = "multi-repair row, highest logconf"
            else:
                f = max(pool, key=lambda f: max(h[3] for h in f['gr']))
                h = max(f['gr'], key=lambda h: h[3])
                verdict = (f['r'], h[0], h[1])
                why = "generalized repair (d7 remapped to %d)" % h[2]
        else:
            f = max(fails, key=lambda f: f['tension']) if fails else None
            if f is not None and f['legal_row'] is not None:
                lr = f['legal_row']
                p_ = next(j for j in range(7) if lr[j] != f['nat'][j])
                verdict = (f['r'], p_, lr[p_])
                why = "fallback: highest-tension row via best legal parse"
            else:
                verdict = (0, 5, 0); why = 'degenerate fallback'
    digits = D.copy()
    for r in range(64):
        digits[8 * r + 1] = (r + 1) // 10
        digits[8 * r + 2] = (r + 1) % 10
        digits[8 * r + 0] = nat_of[r][0]
    for f in fails:
        r = f['r']
        if verdict is not None and r == verdict[0]:
            for j in range(8):
                digits[8 * r + j] = f['nat'][j]
            digits[8 * r + 1] = (r + 1) // 10
            digits[8 * r + 2] = (r + 1) % 10
        elif f['legal_row'] is not None:
            for j in range(8):
                digits[8 * r + j] = f['legal_row'][j]
    ledger = "R%02d:%d:%d" % verdict
    if verbose:
        print(f"--- VERDICT {ledger}  ({why}) ---")
    return digits, ledger, fails, nat_of
