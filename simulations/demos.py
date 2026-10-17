"""Toy metadynamics demonstrations (overdamped Langevin, kT = 1, D = 1).
Demo 1: standard metaD filling a 3-well profile (cf. Laio & Gervasio 2008; Barducci et al. 2011, Fig. 1)
Demo 2: standard vs well-tempered convergence of the bias
Demo 3: infrequent metaD kinetics vs brute-force MD (cf. Ray & Parrinello 2023, eqs. 11-14, Fig. 2)
"""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratestats import bias_at, fit_tau, calibrated_ks, kaminsky_ci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 160, "font.family": "DejaVu Sans"})
INK, ACC, ACC2, GREY = "#1b2433", "#d9622b", "#2f6db5", "#8a93a3"
plt.rcParams.update({"figure.facecolor":"#ffffff","axes.facecolor":"#ffffff","font.size":15})

# ---------- 3-well profile (Laio & Gervasio Fig 5d form), reflecting walls at +-4
def F3(x):
    return (-5*np.exp(-((x-2)/0.75)**2) - 4*np.exp(-(x/0.75)**2) - 7*np.exp(-((x+2)/0.75)**2))
def dF3(x, h=1e-4):
    return (F3(x+h)-F3(x-h))/(2*h)

grid = np.linspace(-4, 4, 801)

def run_metad(nsteps, w, sigma, tau, dT=None, x0=0.0, dt=1e-3, D=1.0, record_every=None):
    """Bias on a grid; dT=None -> standard metaD, else well-tempered with kB*dT (kT=1)."""
    V = np.zeros_like(grid); dg = grid[1]-grid[0]
    x = x0; traj = []; snaps = []
    for i in range(nsteps):
        if i % tau == 0:
            h = w if dT is None else w*np.exp(-np.interp(x, grid, V)/dT)
            V += h*np.exp(-(grid-x)**2/(2*sigma**2))
        dV = np.interp(x, grid, np.gradient(V, dg)) if i % 1 == 0 else 0
        x += -D*(dF3(x)+dV)*dt + np.sqrt(2*D*dt)*rng.normal()
        if x > 4: x = 8-x
        if x < -4: x = -8-x
        if i % 50 == 0: traj.append((i*dt, x))
        if record_every and i % record_every == 0: snaps.append((i*dt, V.copy()))
    return np.array(traj), V, snaps

# ---------- Demo 1: filling
traj, V, snaps = run_metad(60000, w=0.15, sigma=0.25, tau=100, x0=0.0, record_every=2000)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax[0].plot(traj[:, 0], traj[:, 1], lw=0.6, color=INK)
ax[0].set_xlabel("time"); ax[0].set_ylabel("CV  s(t)")
ax[0].set_title("CV trajectory under metadynamics bias")
for xm, lab in [(-2, "A"), (0, "B"), (2, "C")]:
    ax[0].axhline(xm, color=GREY, ls=":", lw=1); ax[0].text(traj[-1, 0]*1.01, xm, lab, va="center")
Fs = F3(grid)-F3(grid).min()
ax[1].plot(grid, Fs, color=INK, lw=3, label="F(s)")
cols = plt.cm.viridis(np.linspace(0.15, 0.9, 5))
pick = [0, 3, 6, 10, 16]
for c, k in zip(cols, pick):
    t, Vk = snaps[k]
    ax[1].plot(grid, Fs+Vk, color=c, lw=1.4, label=f"F+V$_G$, t={t:.0f}")
ax[1].set_xlabel("s"); ax[1].set_ylabel("free energy (k$_B$T)")
ax[1].set_title("F + V$_G$: Gaussians fill the wells"); ax[1].legend(fontsize=11, frameon=False, loc="upper center", ncol=3); ax[1].set_ylim(-0.5, 13.5)
plt.tight_layout(); plt.savefig("demo1_filling.png"); plt.close()

# ---------- Demo 2: standard vs well-tempered: error of -V vs F over time
def err_curve(dT, nsteps=500000):
    _, _, sn = run_metad(nsteps, w=0.15, sigma=0.25, tau=100, x0=0.0, dT=dT, record_every=10000)
    mask = (grid > -3.2) & (grid < 3.2)
    out = []
    for t, Vk in sn:
        if dT is None: est = -Vk
        else: est = -(1+1/dT)*Vk         # F = -(T+dT)/dT V
        d = (est-F3(grid))[mask]; d -= d.mean()
        out.append((t, np.sqrt((d**2).mean())))
    return np.array(out), sn[-1][1]
e_std, V_std = err_curve(None)
e_wt, V_wt = err_curve(5.0)       # bias factor gamma = (T+dT)/T = 6
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(e_std[:, 0], e_std[:, 1], color=ACC, lw=2, label="standard metaD")
ax[0].plot(e_wt[:, 0], e_wt[:, 1], color=ACC2, lw=2, label="well-tempered, γ = 6")
ax[0].set_xlabel("time"); ax[0].set_ylabel("RMS error of F estimate (k$_B$T)")
ax[0].set_title("Single-run convergence"); ax[0].legend(frameon=False); ax[0].set_ylim(0, None)
ax[1].plot(grid, Fs, color=INK, lw=3, label="true F(s)")
a = -V_std; a -= a[(grid > -3.2) & (grid < 3.2)].min()
b = -(1+1/5.0)*V_wt; b -= b[(grid > -3.2) & (grid < 3.2)].min()
ax[1].plot(grid, a, color=ACC, lw=1.5, label="−V$_G$ (standard)")
ax[1].plot(grid, b, color=ACC2, lw=1.5, ls="--", label="−(T+ΔT)/ΔT · V (WT)")
ax[1].set_xlim(-3.2, 3.2); ax[1].set_ylim(-0.5, 9)
ax[1].set_xlabel("s"); ax[1].set_ylabel("k$_B$T"); ax[1].set_title("Reconstructed profiles, end of run")
ax[1].legend(frameon=False, fontsize=10)
plt.tight_layout(); plt.savefig("demo2_welltempered.png"); plt.close()

# ---------- Demo 3: infrequent metaD kinetics
h = 8.0
def dF2(x): return 4*h*x*(x**2-1)       # F = h (x^2-1)^2, barrier = h = 8 kT
dt = 1e-3
# brute force, vectorised walkers
N = 300; x = -np.ones(N); t_fp = np.full(N, np.nan); alive = np.ones(N, bool); step = 0
while alive.any() and step < 8_000_000:
    xa = x[alive]
    xa += -dF2(xa)*dt + np.sqrt(2*dt)*rng.normal(size=xa.size)
    x[alive] = xa; step += 1
    done = alive & (x > 0.8); t_fp[done] = step*dt; alive &= ~done
tau_md = np.nanmean(t_fp)

# infrequent metaD, vectorised walkers with own bias grids
g2 = np.linspace(-2, 2, 801); dg2 = g2[1]-g2[0]
M = 40; xs = -np.ones(M); Vb = np.zeros((M, g2.size)); tres = np.zeros(M); tb = np.zeros(M)
alive = np.ones(M, bool); w, sig, tin = 0.2, 0.12, 1000   # deposit every 1.0 time unit
step = 0
while alive.any():
    if step % tin == 0:
        idx = np.where(alive)[0]
        Vb[idx] += w*np.exp(-(g2[None, :]-xs[idx, None])**2/(2*sig**2))
    idx = np.where(alive)[0]
    Vloc, dV = bias_at(Vb[idx], xs[idx], g2[0], dg2)   # linear interpolation on the bias grid
    tres[idx] += dt*np.exp(Vloc)                      # JCTC eq. 11: tau = sum dt exp(beta V)
    tb[idx] += dt
    xs[idx] += -(dF2(xs[idx])+dV)*dt + np.sqrt(2*dt)*rng.normal(size=idx.size)
    done = alive & (xs > 0.8); alive &= ~done; step += 1
tau_inmetad = tres.mean(); alpha = (tres/tb).mean()

def ecdf(v):
    v = np.sort(v); return v, np.arange(1, v.size+1)/v.size
tt, cc = ecdf(tres)
tau_c = fit_tau(tres)                    # least-squares fit of 1 - exp(-t/tau_c) to the ECDF
ks_D, p_ks = calibrated_ks(tres)         # KS p-value calibrated by parametric bootstrap
n = tres.size
ci = kaminsky_ci(tres)                   # 95% CI of the mean, the exponential MLE

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
tgrid = np.logspace(np.log10(tt.min()*0.5), np.log10(tt.max()*3), 300)
ax[0].step(tt, cc, where="post", color=INK, lw=2, label=f"rescaled InMetaD times (n={n})")
ax[0].plot(tgrid, 1-np.exp(-tgrid/tau_c), color=ACC2, lw=2, label=f"Poisson fit, τ$_c$ = {tau_c:.0f}")
tm, cm = ecdf(t_fp)
ax[0].step(tm, cm, where="post", color=GREY, lw=1.5, ls="--", label=f"brute-force MD, ⟨τ⟩ = {tau_md:.0f}")
ax[0].set_xscale("log"); ax[0].set_xlabel("first-passage time"); ax[0].set_ylabel("cumulative probability")
ax[0].set_title(f"mean τ = {tres.mean():.0f} (95% CI {ci[0]:.0f}–{ci[1]:.0f}), calibrated KS p = {p_ks:.2f}", fontsize=12); ax[0].legend(frameon=False, fontsize=10, loc="upper left")
ax[1].hist(tres/tb, bins=12, color=ACC2, alpha=0.8)
ax[1].set_xlabel("acceleration factor α = τ / τ$_b$"); ax[1].set_ylabel("count")
ax[1].set_title(f"Speed-up per run (mean α ≈ {alpha:.0f}×)")
plt.tight_layout(); plt.savefig("demo3_kinetics.png"); plt.close()

json.dump({"tau_md": tau_md, "n_md": int(np.isfinite(t_fp).sum()), "tau_inmetad_mean": tau_inmetad,
           "tau_c_fit": tau_c, "p_ks": p_ks, "ci95": ci, "alpha_mean": alpha, "tb_mean": tb.mean(),
           "err_std_final": e_std[-1, 1], "err_wt_final": e_wt[-1, 1]},
          open("demo_results.json", "w"), indent=1, default=float)
print(open("demo_results.json").read())
