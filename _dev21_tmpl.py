import numpy as np

X_TRAIN = np.load('X_kannada_MNIST_train.npz')['arr_0'].astype(np.float64) / 255.0
Y_TRAIN = np.load('y_kannada_MNIST_train.npz')['arr_0'].astype(np.int64)
X_DIG = np.load('X_dig_MNIST.npz')['arr_0'].astype(np.float64) / 255.0
Y_DIG = np.load('y_dig_MNIST.npz')['arr_0'].astype(np.int64)
REST = np.load('_rest_arch.npy')

ANCH_IDX = np.array([8 * r + j for r in range(64) for j in (1, 2)])
ANCH_LAB = np.array([(r + 1) // 10 if j == 1 else (r + 1) % 10
                     for r in range(64) for j in (1, 2)])


def com(img):
    tot = img.sum() + 1e-9
    yy, xx = np.mgrid[0:28, 0:28]
    return (yy * img).sum() / tot, (xx * img).sum() / tot


def center(img):
    cy, cx = com(img)
    sy, sx = int(round(13.5 - cy)), int(round(13.5 - cx))
    return np.roll(img, (sy, sx), (0, 1))


XALL = np.concatenate([X_TRAIN, X_DIG])
YALL = np.concatenate([Y_TRAIN, Y_DIG])
CENT = np.stack([center(XALL[YALL == c].mean(0)) for c in range(10)])
CENT = (CENT - CENT.mean((1, 2), keepdims=True)) / (CENT.std((1, 2), keepdims=True) + 1e-9)

SH = [(dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]


def tmpl_scores(imgs):
    S = np.zeros((len(imgs), 10))
    for i, img in enumerate(imgs):
        z = center(img)
        z = (z - z.mean()) / (z.std() + 1e-9)
        best = np.full(10, -2.0)
        for dy, dx in SH:
            zs = np.roll(z, (dy, dx), (0, 1))
            sc = (zs[None] * CENT).sum((1, 2)) / (28 * 28)
            best = np.maximum(best, sc)
        S[i] = best
    return S


SA = tmpl_scores(REST[ANCH_IDX])
acc = (SA.argmax(1) == ANCH_LAB).mean()
print('template anchor acc: %.4f' % acc)
tens, ones = SA[0::2], SA[1::2]
print('  tens %.3f ones %.3f' % ((tens.argmax(1) == ANCH_LAB[0::2]).mean(),
                                   (ones.argmax(1) == ANCH_LAB[1::2]).mean()))

pr = np.load('_probs_all.npz')
for k in ('Pm', 'Pmft'):
    P = pr[k][ANCH_IDX]
    lg = np.log(np.clip(P, 1e-12, 1))
    T = np.log(np.clip((SA - SA.min(1, keepdims=True) + 1e-3), 1e-3, None))
    for w in (0.3, 0.5, 0.7):
        Q = w * lg + (1 - w) * T
        print(k, 'blend w=%.1f anchor acc %.4f' % (w, (Q.argmax(1) == ANCH_LAB).mean()))

ST = tmpl_scores(REST)
np.save('_tmpl_scores.npy', ST)
print('saved template scores', ST.shape)
