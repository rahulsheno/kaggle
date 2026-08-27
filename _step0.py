# ====================== STEP 0 - OFFICIAL CORPUS - run first ======================
import os, io, hashlib, urllib.request
import numpy as np

WANT = {                                     # sha256 of the raw array bytes
    "X_train": "ec528f157e171d9b", "y_train": "6222ecafc85bee75",
    "X_dig":   "2ce1659f85abab67", "y_dig":   "470b073476546add",
}
MIRROR = ("https://raw.githubusercontent.com/vinayprabhu/Kannada_MNIST/"
          "master/data/output_tensors/MNIST_format")
FILES = {"X_train": "X_kannada_MNIST_train", "y_train": "y_kannada_MNIST_train",
         "X_dig":   "X_dig_MNIST",           "y_dig":   "y_dig_MNIST"}


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _from_kaggle():
    import kagglehub                                          # route 1: official Kaggle
    p = kagglehub.competition_download("Kannada-MNIST")
    rd = lambda f: np.loadtxt(os.path.join(p, f), delimiter=",", skiprows=1, dtype=np.int64)
    tr, dg = rd("train.csv"), rd("Dig-MNIST.csv")
    return {"X_train": tr[:, 1:].reshape(-1, 28, 28).astype(np.uint8),
            "y_train": tr[:, 0].astype(np.uint8),
            "X_dig":   dg[:, 1:].reshape(-1, 28, 28).astype(np.uint8),
            "y_dig":   dg[:, 0].astype(np.uint8)}


def _from_mirror():
    out = {}
    for k, stem in FILES.items():                             # route 2: upstream release
        fn = stem + ".npz"
        if not os.path.exists(fn):
            urllib.request.urlretrieve(f"{MIRROR}/{fn}", fn)
        out[k] = np.load(fn)["arr_0"]
    return out


try:
    CORPUS = _from_kaggle(); route = "kaggle competition Kannada-MNIST"
except Exception as e:
    print("kaggle route unavailable (%s); using upstream mirror" % type(e).__name__)
    CORPUS = _from_mirror(); route = "upstream release"

print("source:", route)
ok = True
for k in FILES:
    g = _h(CORPUS[k]); good = (g == WANT[k]); ok &= good
    print(f"  {k:<8} {str(CORPUS[k].shape):<16} {g}  {'OK' if good else 'MISMATCH'}")
assert ok, "corpus integrity check failed - do not proceed"

X_TRAIN = CORPUS["X_train"].astype(np.float64) / 255.0        # 60000 clean, labelled
Y_TRAIN = CORPUS["y_train"].astype(np.int64)
X_DIG   = CORPUS["X_dig"].astype(np.float64) / 255.0          # 10240 relief-staff hand
Y_DIG   = CORPUS["y_dig"].astype(np.int64)
print("\ncorpus ready:", X_TRAIN.shape, X_DIG.shape)