# Re-runs demos.py with the same RNG sequence, recording everything needed for animation.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
import numpy as np, json
from scipy.optimize import curve_fit
from ratestats import bias_at, fit_tau, calibrated_ks, kaminsky_ci
rng = np.random.default_rng(7)
def F3(x): return (-5*np.exp(-((x-2)/0.75)**2) - 4*np.exp(-(x/0.75)**2) - 7*np.exp(-((x+2)/0.75)**2))
def dF3(x, h=1e-4): return (F3(x+h)-F3(x-h))/(2*h)
grid = np.linspace(-4, 4, 801)
def run_metad(nsteps, w, sigma, tau, dT=None, x0=0.0, dt=1e-3, D=1.0, record_every=None, traj_every=50):
    V = np.zeros_like(grid); dg = grid[1]-grid[0]; x = x0; traj = []; snaps = []; hills = []
    for i in range(nsteps):
        if i % tau == 0:
            h = w if dT is None else w*np.exp(-np.interp(x, grid, V)/dT)
            V += h*np.exp(-(grid-x)**2/(2*sigma**2)); hills.append((round(i*dt,4), round(float(x),4), round(float(h),5)))
        dV = np.interp(x, grid, np.gradient(V, dg))
        x += -D*(dF3(x)+dV)*dt + np.sqrt(2*D*dt)*rng.normal()
        if x > 4: x = 8-x
        if x < -4: x = -8-x
        if i % traj_every == 0: traj.append((round(i*dt,3), round(float(x),3)))
        if record_every and i % record_every == 0: snaps.append((i*dt, V.copy()))
    return traj, V, snaps, hills
out = {}
# Demo 1
traj, V, snaps, hills = run_metad(60000, w=0.15, sigma=0.25, tau=100, x0=0.0, record_every=2000)
out["d1"] = {"traj": traj, "hills": hills, "w": 0.15, "sigma": 0.25, "tauG": 0.1}
# Demo 2
mask = (grid > -3.2) & (grid < 3.2)
def run2(dT):
    traj, V, sn, hills = run_metad(500000, w=0.15, sigma=0.25, tau=100, x0=0.0, dT=dT, record_every=2000, traj_every=500)
    err = []
    for t, Vk in sn:
        est = -Vk if dT is None else -(1+1/dT)*Vk
        d = (est-F3(grid))[mask]; d -= d.mean(); err.append((round(t,2), round(float(np.sqrt((d**2).mean())),4)))
    return {"traj": traj, "hills": hills, "err": err}
out["d2"] = {"std": run2(None), "wt": run2(5.0), "dT": 5.0}
print("d2 final", out["d2"]["std"]["err"][-1], out["d2"]["wt"]["err"][-1])
# Demo 3
h = 8.0
def dF2(x): return 4*h*x*(x**2-1)
dt = 1e-3
N = 300; x = -np.ones(N); t_fp = np.full(N, np.nan); alive = np.ones(N, bool); step = 0
while alive.any() and step < 8_000_000:
    xa = x[alive]; xa += -dF2(xa)*dt + np.sqrt(2*dt)*rng.normal(size=xa.size)
    x[alive] = xa; step += 1
    done = alive & (x > 0.8); t_fp[done] = step*dt; alive &= ~done
tau_md = np.nanmean(t_fp)
g2 = np.linspace(-2, 2, 801); dg2 = g2[1]-g2[0]
M = 40; xs = -np.ones(M); Vb = np.zeros((M, g2.size)); tres = np.zeros(M); tb = np.zeros(M)
alive = np.ones(M, bool); w, sig, tin = 0.2, 0.12, 1000
SHOW = [0, 1, 2]
wtraj = {k: [] for k in SHOW}; whills = {k: [] for k in SHOW}; finish = np.zeros(M)
step = 0
while alive.any():
    if step % tin == 0:
        idx = np.where(alive)[0]
        Vb[idx] += w*np.exp(-(g2[None, :]-xs[idx, None])**2/(2*sig**2))
        for k in SHOW:
            if alive[k]: whills[k].append((round(step*dt,3), round(float(xs[k]),4)))
    idx = np.where(alive)[0]
    Vloc, dV = bias_at(Vb[idx], xs[idx], g2[0], dg2)        # linear interpolation on the bias grid
    tres[idx] += dt*np.exp(Vloc); tb[idx] += dt
    xs[idx] += -(dF2(xs[idx])+dV)*dt + np.sqrt(2*dt)*rng.normal(size=idx.size)
    if step % 50 == 0:
        for k in SHOW:
            if alive[k]: wtraj[k].append((round(tb[k],3), round(float(xs[k]),4), round(float(tres[k]),3)))
    done = alive & (xs > 0.8); finish[done] = step; alive &= ~done; step += 1
for k in SHOW: wtraj[k].append((round(tb[k],3), round(float(xs[k]),4), round(float(tres[k]),3)))
tt = np.sort(tres); cc = np.arange(1, M+1)/M
tau_c = fit_tau(tres)                       # least-squares fit to the ECDF
ks_D, p_ks = calibrated_ks(tres)            # calibrated (parametric-bootstrap) KS p-value
ci = kaminsky_ci(tres)                      # 95% CI of the mean (the exponential MLE)
out["d3"] = {"h": h, "w": w, "sigma": sig, "tin": 1.0, "show": {str(k): {"traj": wtraj[k], "hills": whills[k]} for k in SHOW},
  "tres": [round(float(v),2) for v in tres], "tb": [round(float(v),3) for v in tb], "t_fp": [round(float(v),2) for v in t_fp],
  "tau_md": tau_md, "tau_mean": float(tres.mean()), "tau_c": tau_c, "p_ks": p_ks, "ks_D": ks_D, "ci": ci, "ci_md": kaminsky_ci(t_fp), "alpha": float((tres/tb).mean())}
print("d3", tau_md, tau_c, p_ks, ci, out["d3"]["alpha"], [len(wtraj[k]) for k in SHOW], [tres[k] for k in SHOW])
json.dump(out, open(os.path.join(DATA, "sim.json"), "w"), separators=(",", ":"))
