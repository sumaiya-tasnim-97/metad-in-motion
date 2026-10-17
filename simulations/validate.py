"""Numerical checks behind the page's claims. Writes data/validation.json.

    python simulations/validate.py            # all checks (about 10 minutes)
    python simulations/validate.py rates fe   # only some sections

Sections:
  rates     infrequent metaD: calibrated KS test, mean and 95% CI and bias at the barrier top,
            for four independent sets of 200 walkers at each of four hill settings (demo 3)
  fe        free-energy differences over four seeds, standard vs well-tempered (demo 2)
  boundary  ion-pair model: clipped vs reflecting solvent boundary, three seeds (test A)
  interp    bias-grid interpolation against analytic Gaussian derivatives (demos 1-3)
  grid      dipeptide model: periodic seam and grid convergence of the barrier (test B)
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
from ratestats import bias_at, fit_tau, calibrated_ks, kaminsky_ci

OUT = os.path.join(DATA, 'validation.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
want = set(sys.argv[1:]) or {'rates', 'fe', 'boundary', 'interp', 'grid'}

# ---------------------------------------------------------------- demo 3: rates
def inmetad(M, w, sig, tin, seed, h=8.0, dt=1e-3):
    """Vectorized infrequent metaD on F = h (s^2-1)^2, start at -1, stop at s > 0.8."""
    rng = np.random.default_rng(seed); dF = lambda x: 4*h*x*(x**2-1)
    g = np.linspace(-2, 2, 801); dg = g[1]-g[0]; top = np.abs(g) < 0.3
    xs = -np.ones(M); Vb = np.zeros((M, g.size)); tres = np.zeros(M); tb = np.zeros(M)
    alive = np.ones(M, bool); vts = np.zeros(M); step = 0
    while alive.any():
        idx = np.where(alive)[0]
        if step % tin == 0:
            Vb[idx] += w*np.exp(-(g[None, :]-xs[idx, None])**2/(2*sig**2))
        V, dV = bias_at(Vb[idx], xs[idx], g[0], dg)
        tres[idx] += dt*np.exp(V); tb[idx] += dt
        xs[idx] += -(dF(xs[idx])+dV)*dt + np.sqrt(2*dt)*rng.normal(size=idx.size)
        done = alive & (xs > 0.8)
        for k in np.where(done)[0]: vts[k] = Vb[k, top].max()      # bias at the barrier top at escape
        alive &= ~done; step += 1
    return tres, tb, vts

def summarize(t, tb=None, vts=None, B=2000):
    D, p = calibrated_ks(t, B=B); lo, hi = kaminsky_ci(t)
    out = dict(n=int(len(t)), mean=float(np.mean(t)), ci=[float(lo), float(hi)], tau_fit=float(fit_tau(t)), ks_D=float(D), p=float(p))
    if tb is not None: out['alpha'] = float(np.mean(t/tb))
    if vts is not None: out['V_top_mean'] = float(vts.mean()); out['V_top_max'] = float(vts.max())
    return out

if 'rates' in want:
    t0 = time.time(); sim = json.load(open(os.path.join(DATA, 'sim.json')))['d3']
    rates = {'md': dict(label='Brute-force MD', **summarize(np.array(sim['t_fp']))),
             'page': dict(label='Page settings, 40 walkers', **summarize(np.array(sim['tres']), np.array(sim['tb'])))}
    seeds = [21, 22, 23, 24]; rates['seeds'] = seeds
    for key, label, w, tin in [('page', 'Page settings', 0.2, 1000), ('slow', 'Hills half as often', 0.2, 2000),
                               ('small', 'Hills half as high', 0.1, 1000), ('fast', 'Hills 10× too often', 0.2, 100)]:
        runs = []
        for seed in seeds:                                  # four independent sets of 200 walkers
            tr, tb, vts = inmetad(200, w, 0.12, tin, seed)
            runs.append(summarize(tr, tb, vts, B=1000))
        m = np.array([r['mean'] for r in runs])
        rates['set_' + key] = dict(label=label, w=w, stride=tin/1000, runs=runs, mean_of_means=float(m.mean()), sd_of_means=float(m.std(ddof=1)),
                                   n_rejected=int(sum(r['p'] < 0.05 for r in runs)), V_top_mean=float(np.mean([r['V_top_mean'] for r in runs])))
        print('rates', key, np.round(m), 'p', [round(r['p'], 3) for r in runs], flush=True)
    res['rates'] = rates; print('rates done in %.0f s' % (time.time()-t0))

# ---------------------------------------------------------------- demo 2: free-energy recovery
if 'fe' in want:
    t0 = time.time()
    F3 = lambda x: -5*np.exp(-((x-2)/0.75)**2) - 4*np.exp(-(x/0.75)**2) - 7*np.exp(-((x+2)/0.75)**2)
    dF3 = lambda x, h=1e-4: (F3(x+h)-F3(x-h))/(2*h)
    grid = np.linspace(-4, 4, 801); dg = grid[1]-grid[0]
    def run(seed, dT, nsteps=500000, w=0.15, sigma=0.25, tau=100, dt=1e-3):
        rng = np.random.default_rng(seed); V = np.zeros_like(grid); x = 0.0
        for i in range(nsteps):
            if i % tau == 0:
                hgt = w if dT is None else w*np.exp(-np.interp(x, grid, V)/dT)
                V += hgt*np.exp(-(grid-x)**2/(2*sigma**2))
            x += -(dF3(x)+np.interp(x, grid, np.gradient(V, dg)))*dt + np.sqrt(2*dt)*rng.normal()
            if x > 4: x = 8-x
            if x < -4: x = -8-x
        return V
    def obs(F):
        at = lambda s: F[np.argmin(abs(grid-s))]
        return [at(0)-at(-2), at(2)-at(-2), F[(grid > -1.5) & (grid < -0.4)].max()-at(-2)]
    fe = {'names': ['F(B) − F(A)', 'F(C) − F(A)', 'barrier A → B'], 'true': [float(v) for v in obs(F3(grid))], 'seeds': [101, 102, 103, 104]}
    for key, dT in [('standard', None), ('wt', 5.0)]:
        rows = []
        for seed in fe['seeds']:
            V = run(seed, dT); rows.append([float(v) for v in obs(-V if dT is None else -(1+1/dT)*V)])
        a = np.array(rows); fe[key] = dict(rows=rows, mean=a.mean(0).tolist(), sd=a.std(0, ddof=1).tolist())
        print('fe', key, np.round(a.mean(0), 2), np.round(a.std(0, ddof=1), 2), flush=True)
    res['fe'] = fe; print('fe done in %.0f s' % (time.time()-t0))

# ---------------------------------------------------------------- test A: boundary treatment
if 'boundary' in want:
    t0 = time.time(); from surf import kT
    src = open(os.path.join(HERE, 'sim2002.py')).read().split("if __name__")[0]
    refl = "        if y < 0.0: y = -y                     # reflecting walls for the solvent coordinate\n        if y > 1.2: y = 2.4 - y"
    assert refl in src
    variants = {'reflect': src, 'clip': src.replace(refl, '        y = min(max(y, 0.0), 1.2)')}
    r = np.linspace(2.2, 6, 96); yy = np.linspace(0, 1.2, 61); R, Y = np.meshgrid(r, yy, indexing='ij')
    def barrier(H):
        V = np.zeros_like(R)
        for a, b in H: V += 0.3*np.exp(-(R-a)**2/(2*.12**2)-(Y-b)**2/(2*.04**2))
        E = -V; E -= E[r < 3.3].min(); F1 = -kT*np.log(np.exp(-E/kT).sum(1)); F1 -= F1[r < 3.3].min()
        return float(F1[(r > 3.5) & (r < 4.5)].max())
    bnd = {'seeds': [2002, 7, 99]}
    for key, code in variants.items():
        rows = []
        for seed in bnd['seeds']:
            ns = {'__file__': os.path.join(HERE, 'sim2002.py')}; exec(code, ns); ns['rng'] = np.random.default_rng(seed)
            H, T = ns['runA'](); T = np.array(T)
            rows.append(dict(seed=seed, barrier=barrier(H), dissociation=int(next(i for i, (x, y) in enumerate(H) if x > 4.8) + 1),
                             edge_fraction=float((T[:, 1] >= 1.19).mean())))
            print('boundary', key, rows[-1], flush=True)
        bnd[key] = rows
    res['boundary'] = bnd; print('boundary done in %.0f s' % (time.time()-t0))

# ---------------------------------------------------------------- interpolation accuracy
if 'interp' in want:
    rng = np.random.default_rng(3)
    grid = np.linspace(-4, 4, 801); dg = grid[1]-grid[0]; c = rng.uniform(-3, 3, 600); w, s = 0.15, 0.25
    V = (w*np.exp(-(grid[:, None]-c)**2/(2*s*s))).sum(1); xq = rng.uniform(-3.9, 3.9, 4000)
    num = np.interp(xq, grid, np.gradient(V, dg)); ana = (-(w*(xq[:, None]-c)/s**2)*np.exp(-(xq[:, None]-c)**2/(2*s*s))).sum(1)
    g2 = np.linspace(-2, 2, 801); dg2 = g2[1]-g2[0]; c2 = rng.uniform(-1.4, -0.4, 45)
    V2 = (0.2*np.exp(-(g2[:, None]-c2)**2/(2*.12**2))).sum(1); xq2 = rng.uniform(-1.6, 0.9, 4000)
    Vex = (0.2*np.exp(-(xq2[:, None]-c2)**2/(2*.12**2))).sum(1)
    gi = np.clip(((xq2+2)/dg2).astype(int), 1, 799); Vin, _ = bias_at(np.tile(V2, (xq2.size, 1)), xq2, g2[0], dg2)
    res['interp'] = dict(force_rms=float(np.sqrt(((num-ana)**2).mean())), force_max=float(np.abs(num-ana).max()), force_typical=float(np.abs(ana).mean()),
                         nearest_V_max=float(np.abs(V2[gi]-Vex).max()), interp_V_max=float(np.abs(Vin-Vex).max()))
    print('interp', res['interp'])

# ---------------------------------------------------------------- test B: seam and grid convergence
if 'grid' in want:
    from surf import FB, bottleneck
    HB = np.array(json.load(open(os.path.join(DATA, 'raw2002.json')))['B']['hills'])[:700]
    wr = lambda a: (a+180) % 360 - 180; conv = []
    for n in [72, 144, 288]:
        ax = np.linspace(-180, 180, n+1)[:-1]; P, Q = np.meshgrid(ax, ax, indexing='ij'); V = np.zeros_like(P)
        for a, b in HB: V += 0.2*np.exp(-(wr(P-a)**2+wr(Q-b)**2)/(2*14**2))
        out = []
        for G in (-V, FB(P, Q)):
            G = G - G.min(); mn = lambda c: np.unravel_index(np.argmin(np.where(c, G, 99)), G.shape)
            ia = mn((abs(P+30) < 40) & (abs(Q+63) < 40)); ic = mn((P > 120) & (abs(Q+75) < 30))
            out.append(float(bottleneck(G, ia, ic, True) - G[ic]))
        conv.append(dict(spacing=360/n, estimate=out[0], true=out[1]))
    ax = np.linspace(-180, 180, 73); P, Q = np.meshgrid(ax, ax, indexing='ij'); V = np.zeros_like(P)
    for a, b in HB: V += 0.2*np.exp(-(wr(P-a)**2+wr(Q-b)**2)/(2*14**2))
    res['grid'] = dict(convergence=conv, seam_max=float(max(np.abs(V[0]-V[-1]).max(), np.abs(V[:, 0]-V[:, -1]).max())))
    print('grid', res['grid'])

json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT)
