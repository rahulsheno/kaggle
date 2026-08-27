import numpy as np, torch, copy
import torch.nn as nn
exec(open('_dev3_final.py').read().split('N, batch, epochs')[0])
net = Net()
base = torch.load('_denoiser_final.pt')
CIN = torch.from_numpy(np.stack([make_features(z)[0] for z in CN]))
CCT = torch.from_numpy(CC.astype(np.float32))
crit = nn.MSELoss()
idxs = np.arange(24)
rngf = np.random.default_rng(0)
rngf.shuffle(idxs)
folds = idxs.reshape(6, 4)
net.load_state_dict(base)
with torch.no_grad():
    YB = net(CIN).numpy()[:, 0].clip(0, 1)
print("base strip NRMSE:", nrmse(YB, CC))
for steps in (60, 120, 240, 400):
    gains = []
    for f in range(6):
        net.load_state_dict(base)
        held = folds[f]; traink = np.setdiff1d(idxs, held)
        opt = torch.optim.Adam(net.parameters(), lr=3e-5)
        for st in range(steps):
            loss = crit(net(CIN[traink])[:, 0], CCIN[traink])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            yf = net(CIN[held]).numpy()[:, 0].clip(0, 1)
        gains.append((nrmse(YB[held], CC[held]), nrmse(yf, CC[held])))
    g = np.array(gains)
    print(f'ft-steps {steps}: held NRMSE base {g[:,0].mean():.4f} -> ft {g[:,1].mean():.4f}  (wins {(g[:,1] < g[:,0]).sum()}/6)')
