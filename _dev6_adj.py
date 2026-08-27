import numpy as np, torch
import torch.nn.functional as F
exec(open('_dev1_den.py').read().split('if __name__')[0])
exec(open('_dev5_cls2.py').read().split("if __name__")[0])

def restore_all(net_den, Z):
    outs = np.empty((len(Z), 28, 28))
    with torch.no_grad():
        for i, z in enumerate(Z):
            inp, sig = make_features(z)
            outs[i] = net_den(torch.from_numpy(inp)[None])[0, 0].numpy().clip(0, 1)
    return outs

def logits_of(net, X, nch):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xt = torch.from_numpy(X[i:i+64].astype(np.float32))
            if nch == 1:
                xt = xt[:, None]
            out.append(net(xt).numpy())
    return np.concatenate(out)

# load models
den = Net.__new__(Net) if False else None
import importlib
d3 = importlib.import_module('_dev3_final') if False else None

class DenNet(torch.nn.Module):
    def __init__(self, w=56, nblk=7):
        super().__init__()
        layers = [torch.nn.Conv2d(2, w, 3, padding=1), torch.nn.ReLU(inplace=True)]
        for _ in range(nblk):
            layers += [torch.nn.Conv2d(w, w, 3, padding=1), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Conv2d(w, 1, 3, padding=1)]
        self.body = torch.nn.Sequential(*layers)
    def forward(self, x):
        return x[:, :1] + self.body(x)

net_den = DenNet(56, 7)
net_den.load_state_dict(torch.load('_denoiser_final.pt'))
net_den.eval()
net_cc = Cls(1); net_cc.load_state_dict(torch.load('_cls_clean.pt')); net_cc.eval()
net_cn = Cls(2); net_cn.load_state_dict(torch.load('_cls_noisy.pt')); net_cn.eval()

REST = restore_all(net_den, Z)
INP = np.stack([make_features(z)[0] for z in Z])
lg_clean = logits_of(net_cc, REST, 1)
lg_noisy = logits_of(net_cn, INP, 2)
P_clean = torch.softmax(torch.from_numpy(lg_clean), 1).numpy()
P_noisy = torch.softmax(torch.from_numpy(lg_noisy), 1).numpy()
P = (P_clean + P_noisy) / 2

# anchor accuracy
anchor_idx = []; anchor_lab = []
for r in range(64):
    bed = r + 1
    anchor_idx += [8 * r + 1, 8 * r + 2]
    anchor_lab += [bed // 10, bed % 10]
anchor_idx = np.array(anchor_idx); anchor_lab = np.array(anchor_lab)
for name, Pk in [('clean', P_clean), ('noisy', P_noisy), ('combined', P)]:
    acc = (Pk[anchor_idx].argmax(1) == anchor_lab).mean()
    print(f"anchor acc {name}: {acc:.4f}  (errors at {anchor_idx[Pk[anchor_idx].argmax(1) != anchor_lab]})")

D = P.argmax(1)
rows = D.reshape(64, 8)

def ward_constrained_row(r):
    p = P[8 * r: 8 * r + 8]
    bed = r + 1
    d0 = int(np.array([1, 2, 3, 4])[np.argmax(p[0, [1, 2, 3, 4]])])
    return [d0, bed // 10, bed % 10] + [int(p[j].argmax()) for j in range(3, 8)]

fails = []
for r in range(64):
    nat = ward_constrained_row(r)
    legal = row_is_legal(r, nat)
    closes = check_digit(nat[:7]) == nat[7]
    if legal and closes:
        continue
    repairs = []
    for pos in (0, 3, 4, 5, 6):
        body = nat[:7].copy()
        s = sum(WEIGHTS[j] * body[j] for j in range(7) if j != pos)
        v = ((nat[7] - s) % 10) * INV[WEIGHTS[pos]] % 10
        if v == body[pos]:
            continue
        body[pos] = v
        if pos == 0 and v not in WARDS:
            continue
        if row_is_legal(r, body + [nat[7]]):
            repairs.append((pos, v, float(P[8 * r + pos, nat[pos]])))
    conf = float(np.prod([P[8 * r + j, nat[j]] for j in range(8)]))
    minp = float(min(P[8 * r + j, nat[j]] for j in range(8)))
    fails.append((r, nat, legal, closes, repairs, conf, minp))
    print(f"FAIL row {r:2d} read={nat} legal={legal} closes={closes} repairs={[(p_, v_, round(c, 3)) for p_, v_, c in repairs]} conf={conf:.4f} minp={minp:.3f}")

print()
print("=== verdict candidates (exactly one repair) ===")
cand = [f for f in fails if len(f[4]) == 1]
cand.sort(key=lambda f: -f[5])
for r, nat, legal, closes, repairs, conf, minp in cand:
    p_, v_, c = repairs[0]
    print(f"R{r:02d}:{p_}:{v_}  (read digit at pos {p_} was {nat[p_]} with conf {c:.3f}; row conf {conf:.4f}, minp {minp:.3f})")
np.savez('_stage1.npz', REST=REST, P=P, lg_clean=lg_clean, lg_noisy=lg_noisy)
print('stage1 saved')
