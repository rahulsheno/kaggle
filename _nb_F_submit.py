# ============================ SUBMISSION - run last ============================
def write_submission(restored, digits, ledger, path="submission.npz",
                     csv_path="submission.csv"):
    import re
    R = np.asarray(restored, dtype=np.float32)
    D = np.asarray(digits).astype(np.int64).reshape(-1)
    L = str(ledger).strip().upper()

    assert R.shape == (512, 28, 28), f"restored must be (512,28,28), got {R.shape}"
    assert np.isfinite(R).all(),      "restored contains NaN or inf"
    assert D.shape == (512,),         f"digits must be (512,), got {D.shape}"
    assert D.min() >= 0 and D.max() <= 9, "digits must lie in 0..9"
    m = re.fullmatch(r"R(\d{2}):([0-6]):(\d)", L)
    assert m, 'ledger must look like "R07:5:3"'
    assert 0 <= int(m.group(1)) <= 63, "row out of range"

    out = float(R.min()), float(R.max())
    if out[0] < 0 or out[1] > 1:
        print("warning: restored outside [0,1] (%.3f, %.3f); it will be clipped" % out)

    R = np.clip(R, 0, 1)
    np.savez_compressed(path, restored=R, digits=D, ledger=L)
    print("wrote", path)

    lrow, lpos, ldig = int(m.group(1)), int(m.group(2)), int(m.group(3))
    lines = ["Id,Value"]
    for c in range(512):
        px = R[c].ravel()
        lines += ["P_%03d_%03d,%.6f" % (c, k, v) for k, v in enumerate(px)]
    lines += ["D_%03d,%d" % (c, d) for c, d in enumerate(D)]
    lines += ["L_ROW,%d" % lrow, "L_POS,%d" % lpos, "L_DIG,%d" % ldig,
              "LP_ROW,%d" % lrow, "LP_POS,%d" % lpos, "LP_DIG,%d" % ldig]
    with open(csv_path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")
    assert len(lines) == 1 + 512 * 784 + 512 + 6, "row count must be 401,926"
    print("wrote", csv_path, "(%d data rows)" % (len(lines) - 1))
    print("  restored  ", R.shape, "mean %.4f" % R.mean())
    print("  digits    ", np.bincount(D, minlength=10))
    print("  ledger    ", L)


write_submission(restored, digits, ledger)
