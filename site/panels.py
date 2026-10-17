import html, re, os, json
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_j = lambda f: json.load(open(os.path.join(ROOT, 'data', f)))
SIM, S2, VAL = _j('sim.json'), _j('sim2002.json'), _j('validation.json')
D3, RT, FE, BD, IP, GR = SIM['d3'], VAL['rates'], VAL['fe'], VAL['boundary'], VAL['interp'], VAL['grid']
f0 = lambda v: f"{v:.0f}"; f2 = lambda v: f"{v:.2f}"
def table(head, rows):
    return ('<div class="tw"><table class="pt"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>')

def code(s):
    return '<pre class="code"><code>' + html.escape(s.strip('\n')) + '</code></pre>'

def m(k):
    return f'<div class="m" data-m="{k}"></div>'

def det(title, body, tag):
    return f'<details class="x"><summary><span class="xt">{tag}</span>{title}</summary><div class="xb">{body}</div></details>'

def ref(*ns):
    return '<span class="cite">[' + ', '.join(f'<a href="#ref{n}">{n}</a>' for n in ns) + ']</span>'

P = {}

# ------------------------------------------------------------------ Demo 1
P['d1'] = [
det('The model and its parameters', f'''
<p>One coordinate s performs overdamped Langevin dynamics on a three-well profile, the one used in the toy example of Laio &amp; Gervasio (2008) {ref(2)}. All energies are in units of k<sub>B</sub>T, so β = 1.</p>
{m('x_F3')}
<table class="pt"><thead><tr><th>Quantity</th><th>Code</th><th>Value</th></tr></thead><tbody>
<tr><td>Hill height w</td><td><code>w=0.15</code></td><td>0.15 k<sub>B</sub>T</td></tr>
<tr><td>Hill width δs</td><td><code>sigma=0.25</code></td><td>0.25</td></tr>
<tr><td>Deposition stride τ<sub>G</sub></td><td><code>tau=100</code> steps × <code>dt=1e-3</code></td><td>0.1 time units</td></tr>
<tr><td>Diffusion coefficient D</td><td><code>D=1.0</code></td><td>1</td></tr>
<tr><td>Run length</td><td><code>nsteps=60000</code></td><td>60 time units, 600 hills</td></tr>
<tr><td>Bias grid</td><td><code>grid = linspace(-4, 4, 801)</code></td><td>spacing Δg = 0.01</td></tr>
<tr><td>Start</td><td><code>x0=0.0</code></td><td>well B</td></tr>
</tbody></table>
<p>The wells sit at s ≈ −2 (A, F = 0), 0 (B, F = 3.0) and +2 (C, F = 2.0). The A–B barrier is at s ≈ −0.89 (F = 5.24) and the B–C barrier at s ≈ 0.96 (F = 5.50). Beyond |s| ≈ 3.5 the profile is flat at F ≈ 7.</p>
''', 'Model'),
det('The code, line by line', f'''
<p>Everything happens in one function, <code>run_metad</code> in <code>demos.py</code>. The bias is stored on a grid rather than as a list of Gaussians, so the cost per step does not grow with the number of hills.</p>
{code("""
def run_metad(nsteps, w, sigma, tau, dT=None, x0=0.0, dt=1e-3, D=1.0, record_every=None):
    V = np.zeros_like(grid); dg = grid[1]-grid[0]; x = x0; traj = []; snaps = []
    for i in range(nsteps):
        if i % tau == 0:                                   # (1) deposit a hill
            h = w if dT is None else w*np.exp(-np.interp(x, grid, V)/dT)
            V += h*np.exp(-(grid-x)**2/(2*sigma**2))
        dV = np.interp(x, grid, np.gradient(V, dg))         # (2) bias force
        x += -D*(dF3(x)+dV)*dt + np.sqrt(2*D*dt)*rng.normal()   # (3) Langevin step
        if x > 4: x = 8-x                                  # (4) reflecting walls
        if x < -4: x = -8-x
        if i % 50 == 0: traj.append((i*dt, x))
        if record_every and i % record_every == 0: snaps.append((i*dt, V.copy()))
    return np.array(traj), V, snaps
""")}
<p><b>(1) Deposit a hill.</b> Every <code>tau</code> steps a Gaussian is added to the whole grid at once, centred on the current position. In demo 1 <code>dT</code> is <code>None</code>, so the height is always w. This is the discrete form of the sum on the equation card (WIREs eq. 1.2 {ref(3)}).</p>
{m('x_dep')}
<p><b>(2) Bias force.</b> <code>np.gradient</code> takes a centred finite difference of V on the grid, and <code>np.interp</code> reads it off at the walker's position by linear interpolation.</p>
{m('x_grad')}
<p><b>(3) Langevin step.</b> This is the Euler–Maruyama discretization of overdamped Langevin dynamics. The physical force <code>dF3(x)</code> is itself a centred difference with h = 10⁻⁴. With k<sub>B</sub>T = 1 the noise amplitude is √(2DΔt).</p>
{m('x_em')}
<p><b>(4) Walls.</b> A walker that steps past ±4 is mirrored back inside, so it never leaves the grid where the bias is defined.</p>
{m('x_refl')}
<p><b>For the animation</b> the same function was re-run in <code>record.py</code> with the same seed (<code>default_rng(7)</code>). The only change is that every hill's time, centre and height are saved. Your browser rebuilds V<sub>G</sub> from that hill list frame by frame, so nothing is re-simulated or smoothed.</p>
''', 'Code'),
det('What you should see, and why', f'''
<ul>
<li><b>The walker leaves B almost at once.</b> B is a shallow well (3 k<sub>B</sub>T above A) and the barrier to A is only 2.2 k<sub>B</sub>T high from B's side, so thermal noise plus the first few hills push it into A within about 0.3 time units.</li>
<li><b>A takes about 6 time units to fill.</b> A rough count: the volume of well A below the A–B barrier, divided by the volume of one Gaussian.
{m('x_fill')}
58 hills is 5.8 time units of deposition. In the run the walker escapes A at t ≈ 8.5, a little later, because some hills also landed in B and on the walls of A.</li>
<li><b>Then C, then free diffusion.</b> Flattening the whole region |s| &lt; 3.5 to the level of the plateau takes about 220 hills, or 22 time units. After that, F + V<sub>G</sub> is flat to within a few hills and the walker crosses freely.</li>
<li><b>The bias keeps growing after the profile is flat.</b> By t = 60 the maximum of V<sub>G</sub> is 11.8 k<sub>B</sub>T, well above the depth of F. This is expected: once F + V<sub>G</sub> is flat, every new hill lifts the whole profile by roughly the same amount. Only the shape of V<sub>G</sub> carries information. The constant C in lim V<sub>G</sub> = −F + C grows forever.</li>
<li><b>The error does not shrink to zero.</b> For standard metadynamics the residual error scales with the deposition rate {ref(3, 6)}:
{m('x_omega')}
So bumps of order one hill height remain however long you run. Demo 2 shows this directly.</li>
</ul>
''', 'Expect'),
det('Pitfalls', f'''
<ul>
<li><b>Hills too big or too frequent.</b> If w/τ<sub>G</sub> is large, the bias is built faster than the walker can relax in the well. The profile then overfills and the free energy is wrong even though the walker moves. Halving w roughly halves ε² (WIREs eq. 1.6 {ref(3)}).</li>
<li><b>Hills too wide.</b> If δs is comparable to the width of a well, narrow features such as barrier tops are smeared out and barrier heights come out too low. A common rule is to set δs to a fraction of the CV's natural fluctuation in a basin.</li>
<li><b>The time step.</b> With Δt = 10⁻³ and the steepest force in this model (|F′| ≈ 8), a deterministic step moves the walker about 0.008, just under the grid spacing of 0.01. Raising Δt by 10× would make the integrator jump over grid points and bias features.</li>
<li><b>Grid edges.</b> The bias is only defined on [−4, 4]. Without the reflecting walls, a walker pushed out by the bias would see no force from V<sub>G</sub> and could escape to infinity. Real codes such as PLUMED need a bias grid that covers the whole range the CV can reach.</li>
<li><b>Reading the wrong curve.</b> The quantity to compare with F is −V<sub>G</sub> up to a constant. The curve F + V<sub>G</sub> on the plot is a diagnostic: when it is flat, the bias has converged in shape.</li>
<li><b>One CV is a luxury.</b> Here s is the only coordinate, so it is automatically a perfect CV. In real systems a missing slow coordinate causes hysteresis: the bias fills a well, but the system does not leave it the way the CV predicts.</li>
</ul>
''', 'Pitfalls'),
det('Numerical checks', f'''
<p>These come from <code>validate.py</code> (section <code>interp</code>).</p>
<ul>
<li><b>Bias force on the grid.</b> At 4000 random points, the grid force <code>np.interp(x, grid, np.gradient(V))</code> differs from the analytic derivative of 600 Gaussians by {IP['force_rms']:.4f} k<sub>B</sub>T per unit length (rms; worst case {IP['force_max']:.4f}). The typical force is {IP['force_typical']:.1f}, so the error is below 0.1%.</li>
<li><b>Reflecting walls.</b> One deterministic step moves the walker by about 0.008 and the noise by about 0.045, against a domain 8 units wide, so a single reflection is always enough at this time step.</li>
<li><b>The recording matches the run.</b> Each frame shows exactly the hills deposited up to that frame's time: frame k uses hills 1 to k + 1, and the clock reads the time of the last hill.</li>
</ul>
''', 'Checks'),
]

# ------------------------------------------------------------------ Demo 2
P['d2'] = [
det('The model and its parameters', f'''
<p>Same three-well profile, same hills (w = 0.15 k<sub>B</sub>T, δs = 0.25, one every 0.1 time units), and the same integrator as demo 1. Two independent runs of 500 000 steps (500 time units, 5000 hills each):</p>
<ul>
<li><b>Standard</b>: <code>dT=None</code>. Every hill has height w.</li>
<li><b>Well-tempered</b>: <code>dT=5.0</code>, so ΔT = 5T and the bias factor γ = (T + ΔT)/T = 6 {ref(5, 3)}.</li>
</ul>
''', 'Model'),
det('The code, line by line', f'''
<p>The only difference between the runs is the hill height, computed in the deposition branch of <code>run_metad</code>:</p>
{code("""
h = w if dT is None else w*np.exp(-np.interp(x, grid, V)/dT)
V += h*np.exp(-(grid-x)**2/(2*sigma**2))
""")}
{m('x_wtW')}
<p>The current bias under the walker is read off the grid with <code>np.interp</code>. Where the bias is already high, the new hill is exponentially smaller. This is WIREs eq. 1.9 {ref(3)}.</p>
<p>The error curve is computed by <code>err_curve</code>:</p>
{code("""
def err_curve(dT, nsteps=500000):
    _, _, sn = run_metad(nsteps, w=0.15, sigma=0.25, tau=100, x0=0.0, dT=dT, record_every=10000)
    mask = (grid > -3.2) & (grid < 3.2)
    out = []
    for t, Vk in sn:
        if dT is None: est = -Vk
        else: est = -(1+1/dT)*Vk          # F = -(T+dT)/dT V
        d = (est-F3(grid))[mask]; d -= d.mean()
        out.append((t, np.sqrt((d**2).mean())))
    return np.array(out), sn[-1][1]
""")}
<p><b>The estimator.</b> Standard metadynamics uses −V<sub>G</sub>. Well-tempered metadynamics must be rescaled, because its bias converges to only a fraction of −F (WIREs eq. 1.10):</p>
{m('x_wtF')}
<p><b>The error.</b> The constant C is irrelevant, so the code subtracts the mean difference before taking the root-mean-square over the region |s| &lt; 3.2 that both runs sample:</p>
{m('x_rms')}
<p>The animation records a snapshot every 2000 steps (every 2 time units), so the last point is at t = 498. The static plot made by <code>demos.py</code> samples every 10 000 steps instead, so its final values (at t = 490) differ slightly.</p>
''', 'Code'),
det('What you should see, and why', f'''
<ul>
<li><b>The first 20 time units look identical.</b> V is still small, so exp(−V/5) ≈ 1 and the well-tempered hills are nearly full height.</li>
<li><b>Well-tempered hill heights fall steadily</b> (right plot). Once the wells are filled to about 5 k<sub>B</sub>T, new hills are e<sup>−1</sup> ≈ 37% of w, and by the end 83% of new hills are below 0.02 k<sub>B</sub>T. The bias stops changing, so the estimate settles.</li>
<li><b>The standard error never settles.</b> It jumps between about 0.3 and 2 k<sub>B</sub>T for the whole run. This is ε ∝ √(ω/Dβ) in action: the error depends on the deposition rate, not on the run length.</li>
<li><b>What the well-tempered run samples.</b> Because F + V → F/γ, the CV is sampled as if it were at temperature γT = 6T (WIREs eq. 1.11), while all other degrees of freedom stay at T:
{m('x_P11')}
With F spanning 7 k<sub>B</sub>T, the effective barriers become 7/6 ≈ 1.2 k<sub>B</sub>T, which is why the WT walker still visits every well.</li>
<li><b>Final numbers.</b> At t = 498: 1.73 k<sub>B</sub>T standard against 0.31 k<sub>B</sub>T well-tempered. Because the standard error fluctuates, a snapshot a few time units earlier gives different values (at t = 490: 1.7 against 0.45).</li>
</ul>
''', 'Expect'),
det('Pitfalls', f'''
<ul>
<li><b>Forgetting the rescaling factor.</b> Reading −V straight off a well-tempered run gives F scaled by ΔT/(T + ΔT) = 5/6, so every barrier comes out 17% too low. For γ = 2 it would be 50% too low.</li>
<li><b>ΔT too small.</b> A bias factor near 1 tempers the hills before the wells are filled. The walker never escapes and the run looks "converged" in a single basin. Choose γ so that γk<sub>B</sub>T is comparable to the highest barrier you need to cross.</li>
<li><b>ΔT too large.</b> As ΔT → ∞ well-tempered becomes standard metadynamics again, and the error stops settling.</li>
<li><b>Judging convergence by one number.</b> A single RMS error hides where the error sits. Always check that the estimate stops changing at the barrier tops, which are sampled least.</li>
<li><b>Using the true F as a check.</b> This demo can compute ε only because F is known. In real work, use the time evolution of free-energy differences between basins, or block averages over several runs {ref(3)}.</li>
<li><b>Edges.</b> The error is restricted to |s| &lt; 3.2 because near the reflecting walls both runs over-deposit, and that artefact would dominate the RMS.</li>
</ul>
''', 'Pitfalls'),
det('Numerical checks', f'''
<p>Four more independent runs of each kind (seeds {', '.join(map(str, FE['seeds']))}), 500 time units each, from <code>validate.py</code> (section <code>fe</code>). The table compares free-energy differences read off the final bias with the exact values of the model.</p>
{table(['Quantity (k<sub>B</sub>T)', 'Exact', 'Standard: mean ± SD', 'Well-tempered: mean ± SD'],
       [[FE['names'][i], f2(FE['true'][i]), f"{FE['standard']['mean'][i]:.2f} ± {FE['standard']['sd'][i]:.2f}", f"{FE['wt']['mean'][i]:.2f} ± {FE['wt']['sd'][i]:.2f}"] for i in range(3)])}
<ul>
<li><b>Well-tempered is reproducible.</b> Its scatter between seeds is {min(FE['wt']['sd']):.1f}–{max(FE['wt']['sd']):.1f} k<sub>B</sub>T, and its means sit within {max(abs(a-b) for a, b in zip(FE['wt']['mean'], FE['true'])):.1f} k<sub>B</sub>T of the exact values.</li>
<li><b>Standard metadynamics is not.</b> The same quantity can differ by {max(FE['standard']['sd']):.1f} k<sub>B</sub>T (one SD) from one run to the next, because the bias never stops changing.</li>
<li><b>This check needs the exact F.</b> The error curve on the plot is measured against the known model surface. In a real simulation F is unknown, so you would compare independent runs, or follow the free-energy difference between basins over time.</li>
</ul>
''', 'Checks'),
]

# ------------------------------------------------------------------ Demo 3
P['d3'] = [
det('The model and its parameters', f'''
<p>A symmetric double well in one coordinate, with an 8 k<sub>B</sub>T barrier. The particle starts at s = −1 and the "event" is the first passage past s = 0.8.</p>
{m('x_F2')}
<table class="pt"><thead><tr><th>Quantity</th><th>Brute-force MD</th><th>Infrequent metaD</th></tr></thead><tbody>
<tr><td>Independent runs</td><td>N = 300</td><td>M = 40 (and 4 × 200 in the checks)</td></tr>
<tr><td>Hill height / width</td><td>none</td><td>0.2 k<sub>B</sub>T / 0.12</td></tr>
<tr><td>Deposition stride</td><td>none</td><td>1000 steps (1 time unit)</td></tr>
<tr><td>Bias</td><td>none</td><td>standard (not well-tempered)</td></tr>
<tr><td>Time step / D / k<sub>B</sub>T</td><td colspan="2">10⁻³ / 1 / 1</td></tr>
</tbody></table>
''', 'Model'),
det('The code, line by line', f'''
<p><b>Reference: vectorized brute-force MD.</b> All 300 walkers move together. A walker that crosses s = 0.8 records its time and stops.</p>
{code("""
N = 300; x = -np.ones(N); t_fp = np.full(N, np.nan); alive = np.ones(N, bool); step = 0
while alive.any() and step < 8_000_000:
    xa = x[alive]
    xa += -dF2(xa)*dt + np.sqrt(2*dt)*rng.normal(size=xa.size)
    x[alive] = xa; step += 1
    done = alive & (x > 0.8); t_fp[done] = step*dt; alive &= ~done
tau_md = np.nanmean(t_fp)
""")}
<p><b>Infrequent metadynamics.</b> Each of the 40 walkers has its own bias grid <code>Vb[k]</code> on [−2, 2] with spacing 0.005. Two clocks run side by side: <code>tb</code> counts simulated time, and <code>tres</code> accumulates the rescaled time.</p>
{code("""
while alive.any():
    if step % tin == 0:                                      # one hill per time unit
        idx = np.where(alive)[0]
        Vb[idx] += w*np.exp(-(g2[None, :]-xs[idx, None])**2/(2*sig**2))
    idx = np.where(alive)[0]
    Vloc, dV = bias_at(Vb[idx], xs[idx], g2[0], dg2)        # linear interpolation on the bias grid
    tres[idx] += dt*np.exp(Vloc)                             # JCTC eq. 11
    tb[idx] += dt
    xs[idx] += -(dF2(xs[idx])+dV)*dt + np.sqrt(2*dt)*rng.normal(size=idx.size)
    done = alive & (xs > 0.8); alive &= ~done; step += 1
""")}
<p>The rescaling is a running sum over every step of the biased run (JCTC eq. 11 {ref(4)}, from Tiwary &amp; Parrinello {ref(9)}):</p>
{m('x_resc')}
<p><b>Analysis.</b> Three numbers come out of the 40 rescaled times, all from <code>ratestats.py</code>:</p>
{code("""
tau_c = fit_tau(tres)              # least-squares fit of 1 - exp(-t/tau_c) to the empirical CDF
ks_D, p_ks = calibrated_ks(tres)   # KS distance to the fitted exponential, bootstrap-calibrated p
ci = kaminsky_ci(tres)             # 95% chi-squared interval for the mean
""")}
{m('x_fit')}
<p><b>The calibrated KS test.</b> The KS distance D measures the largest gap between the data's CDF and the fitted exponential. Because τ<sub>c</sub> was fitted to the same data, D is smaller than it would be against the true distribution, and the textbook KS p-value is too optimistic. The code calibrates it with a parametric bootstrap: draw 40 exponential times, refit τ, recompute D, 2000 times over. The p-value is the fraction of synthetic D at least as large as the observed one. This is the idea behind the Lilliefors test {ref(11)}.</p>
{m('x_ks')}
{code("""
def calibrated_ks(t, B=2000, seed=12345):
    t = np.asarray(t); n = t.size; rng = np.random.default_rng(seed)
    tc = fit_tau(t); D = kstest(t, 'expon', args=(0, tc)).statistic
    Ds = np.empty(B)
    for b in range(B):
        x = rng.exponential(tc, n)
        Ds[b] = kstest(x, 'expon', args=(0, fit_tau(x))).statistic
    return D, float((Ds >= D).mean())
""")}
<p><b>The confidence interval</b> is Kaminsky's χ² interval for an exponential mean (JCTC eq. 14 {ref(4, 12)}). With these numbers:</p>
{m('x_cinum')}
''', 'Code'),
det('What you should see, and why', f'''
<ul>
<li><b>The brute-force answer is predictable.</b> Kramers' formula for overdamped escape gives
{m('x_kram')}
close to the measured {f0(D3['tau_md'])}. The small difference is expected: the event is defined at s = 0.8 rather than at the far minimum, and Kramers' formula assumes a very high barrier.</li>
<li><b>Agreement with brute force.</b> The 40 rescaled times have mean τ = {f0(D3['tau_mean'])} (95% CI {f0(D3['ci'][0])}–{f0(D3['ci'][1])}), against {f0(D3['tau_md'])} (CI {f0(D3['ci_md'][0])}–{f0(D3['ci_md'][1])}) from 300 unbiased runs. The least-squares fit to the CDF gives τ<sub>c</sub> = {f0(D3['tau_c'])}. For exponential data the plain mean is the maximum-likelihood estimate and the quantity the interval belongs to, so it is the number to quote.</li>
<li><b>Exponential statistics.</b> With an 8 k<sub>B</sub>T barrier the walker equilibrates in the well long before escaping, so the times should be memoryless. The calibrated KS test gives p = {D3['p_ks']:.2f}, so the exponential model is not rejected. With only 40 events the test has limited power, which is why the checks below repeat the run with far more walkers.</li>
<li><b>A modest speed-up, about {f0(D3['alpha'])}×.</b> The biased runs last about {np.mean(D3['tb']):.0f} time units, so each walker deposits about that many hills. The speed-up grows exponentially with the bias, so with real barriers of 20–30 k<sub>B</sub>T it can be enormous {ref(4)}.</li>
<li><b>The barrier top is nearly, not perfectly, bias-free.</b> Across the 800 walkers of the checks at these settings, the bias near the barrier top (|s| &lt; 0.3) at the moment of escape averaged {RT['set_page']['V_top_mean']:.2f} k<sub>B</sub>T and never exceeded {max(r['V_top_max'] for r in RT['set_page']['runs']):.1f} k<sub>B</sub>T, small against the 8 k<sub>B</sub>T barrier.</li>
</ul>
''', 'Expect'),
det('Numerical checks', f'''
<p>One run of 40 walkers cannot show whether the method is right or merely lucky. <code>validate.py</code> (section <code>rates</code>) therefore repeats the calculation with four independent sets of 200 walkers at each of four hill settings, and applies the calibrated KS test to each set.</p>
{table(['Setting', 'Hills', 'Mean τ of each set', 'Mean ± SD of the four', 'Sets failing KS (p &lt; 0.05)', 'Bias at barrier top'],
       [['Brute-force MD', 'none', '—', f"{RT['md']['mean']:.0f} (95% CI {RT['md']['ci'][0]:.0f}–{RT['md']['ci'][1]:.0f})", f"p = {RT['md']['p']:.2f}", '0']] +
       [[s['label'], f"{s['w']} k<sub>B</sub>T every {s['stride']:g}", ', '.join(f0(r['mean']) for r in s['runs']), f"{s['mean_of_means']:.0f} ± {s['sd_of_means']:.0f}", f"{s['n_rejected']} of {len(s['runs'])}", f"{s['V_top_mean']:.2f} k<sub>B</sub>T"]
        for s in (RT['set_page'], RT['set_slow'], RT['set_small'], RT['set_fast'])])}
<ul>
<li><b>Within the rules, the rate is right.</b> With hills of 0.2 k<sub>B</sub>T every 1 or 2 time units, or 0.1 k<sub>B</sub>T every 1, the averages of the four sets lie within {max(abs(RT[k]['mean_of_means']/RT['md']['mean']-1) for k in ('set_page','set_slow','set_small'))*100:.0f}% of the brute-force {f0(RT['md']['mean'])}.</li>
<li><b>Breaking the rules gives the wrong rate, and the test notices.</b> With hills every 0.1 time units the bias reaches the barrier top ({RT['set_fast']['V_top_mean']:.2f} k<sub>B</sub>T on average), τ comes out {RT['set_fast']['mean_of_means']/RT['md']['mean']:.1f}× too long ({f0(RT['set_fast']['mean_of_means'])} ± {f0(RT['set_fast']['sd_of_means'])}), and all four sets fail the KS test.</li>
<li><b>The shape test also rejects some good runs.</b> {sum(RT[k]['n_rejected'] for k in ('set_page','set_slow','set_small'))} of the 12 sets with acceptable settings fail at p &lt; 0.05 even though their means are accurate, more than the 1 in 20 expected by chance. With 200 events the test is sensitive enough to see that the rescaled times are only approximately exponential when the bias keeps growing during the wait. Treat a failed test as a reason to look closer, and compare the mean across independent sets.</li>
<li><b>One set is not enough.</b> Single sets of 200 walkers range from {min(min(r['mean'] for r in RT[k]['runs']) for k in ('set_page','set_slow','set_small')):.0f} to {max(max(r['mean'] for r in RT[k]['runs']) for k in ('set_page','set_slow','set_small')):.0f}, consistent with the ±12% width of their 95% intervals. Independent repeats are the honest error bar.</li>
</ul>
''', 'Checks'),
det('Pitfalls', f'''
<ul>
<li><b>Depositing too often.</b> If hills arrive while the walker is near the barrier, bias lands on the transition state and the rescaling no longer describes the real dynamics. In the checks above, hills every 0.1 time units made τ come out {RT['set_fast']['mean_of_means']/RT['md']['mean']:.1f}× too long, and the KS test rejected every set. Try <code>tin = 100</code> to see it.</li>
<li><b>Hills too high.</b> Once the bias approaches the barrier height, escape is no longer a rare event and the exponential assumption fails. Keep the maximum bias clearly below ΔF<sup>‡</sup> (JCTC lists this among the conditions for time rescaling {ref(4)}).</li>
<li><b>A bias that is not quasi-static.</b> JCTC eq. 8 assumes the bias in the basin has settled. Here the bias grows during the whole wait, which is why the rescaled times are only approximately exponential.</li>
<li><b>Grid lookup.</b> The clock multiplies every step by e<sup>βV</sup>, so errors in V go straight into the time. Reading V from the nearest grid cell errs by up to {IP['nearest_V_max']:.2f} k<sub>B</sub>T here, a {(np.exp(IP['nearest_V_max'])-1)*100:.0f}% error in that step's time. The code interpolates linearly instead, which brings the error down to {IP['interp_V_max']:.4f} k<sub>B</sub>T.</li>
<li><b>Too few events, and an uncalibrated test.</b> With fewer than 10 transitions the KS test has almost no power {ref(4)}. A KS p-value computed as if τ<sub>c</sub> were known in advance is too optimistic, because τ<sub>c</sub> was fitted from the same data. Calibrate it, as here, or use the Lilliefors test {ref(11)}.</li>
<li><b>Mean ± SD.</b> For an exponential distribution the standard deviation equals the mean, so "{f0(D3['tau_mean'])} ± {f0(D3['tau_mean'])}" says nothing about how well you know τ. Quote the χ² interval, or the spread over independent sets.</li>
<li><b>Stopping at the first passage.</b> InMetaD needs only forward transitions, but each run must be stopped and checked: did it reach the right product state? A run that wanders into a different state gives a time for the wrong process.</li>
</ul>
''', 'Pitfalls'),
]

# ------------------------------------------------------------------ 2002 A
P['pa'] = [
det('The model and its parameters', f'''
<p>A two-coordinate stand-in for the NaCl system of Laio &amp; Parrinello (2002) {ref(1)}. r is the ion distance in Å and y is a scaled solvent coordinate, plotted as v = 100y to resemble the paper's field axis. The surface is a channel that follows the curve y<sub>p</sub>(r) and rises in v as the ions separate:</p>
{m('x_FA')}
{m('x_U')}
<p>The walls are steep quadratic terms below r = 2.45 Å and above 5.9 Å. Energies are in kcal/mol at 300 K (k<sub>B</sub>T = 0.596). The model's contact pair C is at r = 2.8 Å, the saddle S at r = 4.0 Å with 3.35 kcal/mol, and the separated pair D at r = 5.4 Å with −0.40 kcal/mol.</p>
<table class="pt"><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>
<tr><td>Hill height W</td><td>0.3 kcal/mol (the paper's value)</td></tr>
<tr><td>Widths σ<sub>r</sub>, σ<sub>y</sub></td><td>0.12 Å, 0.04 (4 units of v)</td></tr>
<tr><td>Stride</td><td>one hill every 250 steps, Δt = 4×10⁻⁴</td></tr>
<tr><td>Diffusion</td><td>D<sub>r</sub> = 1, D<sub>y</sub> = 0.12</td></tr>
<tr><td>Length</td><td>1400 hills</td></tr>
</tbody></table>
''', 'Model'),
det('The code, line by line', f'''
<p>The simulation is <code>runA</code> in <code>sim2002.py</code>, and the surface is in <code>surf.py</code>. Here the bias is kept as a list of hill centres, and its force is summed analytically over all of them:</p>
{code("""
for i in range(nhill*tau):
    if i % tau == 0:
        H.append((round(x,4), round(y,4))); hc = np.array(H)
    g = W*np.exp(-(x-hc[:,0])**2/(2*sr*sr) - (y-hc[:,1])**2/(2*sy*sy))
    bx = -(g*(x-hc[:,0])/sr**2).sum(); by = -(g*(y-hc[:,1])/sy**2).sum()   # ∂V_G/∂x, ∂V_G/∂y
    fx, fy = grad(FA, x, y)                                                 # ∂F/∂x, ∂F/∂y
    x += -Dr*(fx+bx)*dt + np.sqrt(2*Dr*kT*dt)*rng.normal()
    y += -Dy*(fy+by)*dt + np.sqrt(2*Dy*kT*dt)*rng.normal()
    if x < 2.2: x = 4.4-x
    if x > 6.0: x = 12-x
    if y < 0.0: y = -y                    # reflecting walls for the solvent coordinate
    if y > 1.2: y = 2.4 - y
""")}
{m('x_lang2')}
<p>The two coordinates get different diffusion coefficients because they have different natural scales: one Å in r against about 0.1 in y.</p>
<p><b>The 1D profile in the bottom panel</b> integrates out the solvent coordinate from the learned 2D map, exactly as the paper did for its 3.4 kcal/mol barrier {ref(1)}:</p>
{m('x_marg')}
<p><code>pre2002.py</code> computes this barrier for every frame (the readout). Your browser recomputes the blue curve from the hills.</p>
''', 'Code'),
det('What you should see, and why', f'''
<ul>
<li><b>The C basin fills first</b>, over about 90 hills, and its contours spread along both axes.</li>
<li><b>Dissociation at hill {S2['A']['diss']+1}.</b> The trail rises in v while r grows. This is the model's version of the paper's point: separating the ions requires reorganizing the solvent.</li>
<li><b>An early 1D estimate is wrong.</b> Just after the first crossing the saddle region has barely been filled, so the 1D barrier reads {next(v for v in S2['A']['bar'] if v is not None):.1f} kcal/mol. Only after many recrossings does it approach the true 3.35.</li>
<li><b>The final estimate</b> wobbles between {min(v for v in S2['A']['bar'][len(S2['A']['bar'])//2:] if v is not None):.1f} and {max(v for v in S2['A']['bar'][len(S2['A']['bar'])//2:] if v is not None):.1f} kcal/mol over the second half of the run and ends at {S2['A']['bar'][-1]:.2f}, against the model's {S2['A']['trueBar']:.2f}. The paper reported 3.4 against 3.3 from thermodynamic integration.</li>
<li><b>The dashed line r = 4.0 Å cuts the channel at an angle.</b> A dividing surface at fixed distance would count some solvent configurations as "crossed" that will fall back. The arrow shows the real direction of the channel through S.</li>
</ul>
''', 'Expect'),
det('Numerical checks', f'''
<p>The solvent coordinate has walls at y = 0 and 1.2. An earlier version of this model clipped y at the walls instead of reflecting it. Clipping is not a valid boundary condition for Langevin dynamics, because it parks the walker exactly on the wall. <code>validate.py</code> (section <code>boundary</code>) runs the model both ways with three seeds.</p>
{table(['Seed', 'Boundary', '1D barrier (kcal/mol)', 'Dissociation at hill', 'Time within 0.01 of the upper wall'],
       [[r['seed'], k, f2(r['barrier']), r['dissociation'], f"{r['edge_fraction']*100:.1f}%"] for k in ('reflect', 'clip') for r in BD[k]])}
<ul>
<li><b>Dissociation is unaffected.</b> It happens at the same hill with either boundary, because the walker only reaches the upper wall late, after the D basin is filled.</li>
<li><b>The barrier changes by up to {max(abs(a['barrier']-b['barrier']) for a, b in zip(BD['reflect'], BD['clip'])):.1f} kcal/mol</b>, about the same as the scatter between seeds ({min(r['barrier'] for r in BD['reflect']):.1f}–{max(r['barrier'] for r in BD['reflect']):.1f} with reflection). The page now uses reflecting walls.</li>
</ul>
''', 'Checks'),
det('Pitfalls', f'''
<ul>
<li><b>This is not the paper's algorithm.</b> The 2002 method moved the CVs by a coarse-grained steepest-descent step using mean forces from short MD runs on 6 replicas {ref(1)}. Here the CVs follow Langevin dynamics directly, which is the later "continuous" metadynamics {ref(3)}. The filling behaviour is the same; the meaning of a "step" is not.</li>
<li><b>Standard metadynamics never settles.</b> With full-height hills of 0.3 kcal/mol ≈ 0.5 k<sub>B</sub>T, the map keeps changing at the level of a hill. Average late frames, or use well-tempered metadynamics.</li>
<li><b>Using only r as the CV.</b> If you bias r alone, the solvent coordinate is left to relax by itself. When it is slow, the bias overfills along r while the system waits for the water: hysteresis, and a barrier that depends on the run.</li>
<li><b>Walls.</b> A coordinate confined to an interval needs reflecting walls. Clipping it to the interval piles probability onto the boundary; see the checks above.</li>
<li><b>Edges of the channel.</b> Regions the walker never visited have no hills. The contour plot leaves them blank, but the marginal F<sub>1</sub>(r) treats them as high free energy, which is correct only if they really are.</li>
</ul>
''', 'Pitfalls'),
]

# ------------------------------------------------------------------ 2002 B
P['pb'] = [
det('The model and its parameters', f'''
<p>A periodic two-angle map shaped like the alanine dipeptide surface in Fig. 3 of the paper {ref(1)}. Each basin is a von Mises well, a periodic analogue of a Gaussian. Two ridges at Φ = 0° and Φ = 130° make the Φ &gt; 0 region hard to reach, as steric clashes do in the real molecule.</p>
{m('x_FB')}
<table class="pt"><thead><tr><th>Basin</th><th>(Ψ, Φ)</th><th>Depth A<sub>k</sub></th><th>F (model)</th></tr></thead><tbody>
<tr><td>α<sub>r</sub></td><td>(−30°, −63°)</td><td>4.6</td><td>0 (global minimum)</td></tr>
<tr><td>C7eq</td><td>(165°, −75°)</td><td>3.15</td><td>1.25</td></tr>
<tr><td>C5</td><td>(165°, −150°)</td><td>2.6</td><td>1.74</td></tr>
<tr><td>α<sub>L</sub></td><td>(40°, 60°)</td><td>1.6</td><td>3.08</td></tr>
</tbody></table>
<p>The C7eq → saddle barrier in the model is 2.11 kcal/mol, which matches the reference 2.1 kcal/mol cited in the paper. Hills are 0.2 kcal/mol high and 14° wide, one every 150 steps, for 700 hills.</p>
''', 'Model'),
det('The code, line by line', f'''
<p>The simulation is <code>runB</code> in <code>sim2002.py</code>. The key difference from every other demo is that the coordinates are angles, so every distance uses the minimum-image convention:</p>
{code("""
def wrap(a): return (a+np.pi) % (2*np.pi) - np.pi
...
dp = wrap(p-hc[:,0]); df = wrap(f-hc[:,1])           # shortest angular distance to each hill
g = W*np.exp(-(dp**2+df**2)/(2*s*s)); bp = -(g*dp).sum()/s**2; bf = -(g*df).sum()/s**2
p = wrap(p - D*(gp+bp)*dt + np.sqrt(2*D*kT*dt)*rng.normal())
f = wrap(f - D*(gf+bf)*dt + np.sqrt(2*D*kT*dt)*rng.normal())
""")}
<p><b>The barrier readout</b> in <code>pre2002.py</code> finds the lowest mountain pass between the C7eq and α<sub>r</sub> minima on the learned map. It uses a minimax (bottleneck) Dijkstra search on the periodic grid: among all paths, take the one whose highest point is lowest.</p>
{m('x_bott')}
{code("""
def bottleneck(G, a, b, periodic):
    best = np.full(G.shape, np.inf); best[a] = G[a]; h = [(G[a], a)]
    while h:
        v, (i, j) = heapq.heappop(h)
        if (i, j) == b: return v                      # height of the lowest pass
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii, jj = (i+di) % n, (j+dj) % m           # periodic neighbours
            nv = max(v, G[ii, jj])
            if nv < best[ii, jj]: best[ii, jj] = nv; heapq.heappush(h, (nv, (ii, jj)))
""")}
''', 'Code'),
det('What you should see, and why', f'''
<ul>
<li><b>C7eq appears on both edges.</b> Ψ = 165° is 15° from the edge, so its hills spill across 180° and reappear at −180°. It is one basin.</li>
<li><b>α<sub>r</sub> is reached at hill 38</b> and needs many more hills than C7eq because it is 1.25 kcal/mol deeper.</li>
<li><b>Until α<sub>r</sub> is full, the map is misleading.</b> The readout "C7eq above α<sub>r</sub>" is negative for a while: the map says C7eq is lower, the opposite of the truth. The order of the basins is only right once both are filled.</li>
<li><b>The barrier estimate wobbles</b> around the true 2.11 kcal/mol, between about 1 and 2.8, and stays near 2.06 from about hill 500 to 600. The paper's 2.0 is a single value of this kind of fluctuating estimate.</li>
<li><b>Φ &gt; 0 is reached late</b> (hill 475), behind the ridge at Φ = 0°. The paper reports the same order of events.</li>
</ul>
''', 'Expect'),
det('Numerical checks', f'''
<p>From <code>validate.py</code> (section <code>grid</code>).</p>
<ul>
<li><b>No seam at ±180°.</b> The bias evaluated at Ψ = −180° and Ψ = +180° (and likewise for Φ) agrees to {GR['seam_max']:.0e} kcal/mol, so the contours join smoothly across the edge of the map.</li>
<li><b>The barrier estimate is grid-converged.</b> The minimax search gives the same pass height on finer grids:</li>
</ul>
{table(['Grid spacing', 'Estimated ΔF‡ from the bias (kcal/mol)', "Model's exact ΔF‡ (kcal/mol)"], [[f"{c['spacing']:g}°", f"{c['estimate']:.3f}", f"{c['true']:.3f}"] for c in GR['convergence']])}
<p>The remaining gap between estimate and exact value ({GR['convergence'][-1]['estimate']-GR['convergence'][-1]['true']:.1f} kcal/mol at hill 700) is the fluctuation of standard metadynamics, not a grid effect.</p>
''', 'Checks'),
det('Pitfalls', f'''
<ul>
<li><b>Forgetting periodicity.</b> Without the minimum-image <code>wrap</code>, a hill at Ψ = 175° would not push a walker at Ψ = −175°, which is 10° away. The bias would build a false wall at the edge and the walker would pile up there.</li>
<li><b>Hill width on angles.</b> 14° is narrow compared with the basins (about 60° across). Much wider hills would merge C7eq and C5, which are only 75° apart in Φ.</li>
<li><b>Reading a barrier too early.</b> Before both basins are filled, the saddle height is measured relative to a wrong baseline. Wait until the walker has crossed back and forth several times.</li>
<li><b>Half the map.</b> The paper's figure shows only Φ &lt; 0. A real run eventually explores Φ &gt; 0 as well. That does not change the earlier barriers, but it does change the normalization if you compute populations.</li>
<li><b>A model, not the molecule.</b> The real dipeptide in water has more slow modes than Ψ and Φ (for example the solvent around the peptide bond). In a real simulation those can make Ψ and Φ an imperfect pair of CVs.</li>
</ul>
''', 'Pitfalls'),
]

ABOUT = det('How this page draws the animations', f'''
<p>Every curve is computed in your browser from recorded simulation data embedded in the page: hill centres and heights, walker positions and, for demo 3, each walker's two clocks.</p>
<ul>
<li><b>The bias</b> at frame k is the sum of the first k recorded Gaussians, evaluated on a grid and accumulated incrementally. Scrubbing backwards rebuilds it from zero.</li>
<li><b>Contour lines</b> on the 2D maps use marching squares. Each grid cell is classified by which of its four corners lie above the contour level, and a line segment is drawn through the interpolated crossing points. Cells with no hill nearby are left blank, which is why unexplored regions are empty.</li>
<li><b>The equations</b> are typeset with MathJax ahead of time and embedded as SVG. The highlighted terms are tagged with classes that the player switches on and off.</li>
<li><b>Files:</b> <code>demos.py</code> (the static figures), <code>record.py</code> (the same runs with full recording), <code>ratestats.py</code> (rate statistics), <code>validate.py</code> (the numerical checks), <code>sim2002.py</code>, <code>surf.py</code> and <code>pre2002.py</code> (the 2002 models).</li>
</ul>
''', 'Page')

REFS = f'''
  <section class="refs" id="refs">
    <p class="eyebrow">References</p>
    <h2>Sources</h2>
    <ol>
      <li id="ref1">A. Laio and M. Parrinello, “Escaping free-energy minima,” <i>Proc. Natl. Acad. Sci. USA</i> <b>99</b>, 12562–12566 (2002). <a href="https://doi.org/10.1073/pnas.202427399">doi:10.1073/pnas.202427399</a> <span class="use">Original metadynamics; test systems A and B.</span></li>
      <li id="ref2">A. Laio and F. L. Gervasio, “Metadynamics: a method to simulate rare events and reconstruct the free energy in biophysics, chemistry and material science,” <i>Rep. Prog. Phys.</i> <b>71</b>, 126601 (2008). <a href="https://doi.org/10.1088/0034-4885/71/12/126601">doi:10.1088/0034-4885/71/12/126601</a> <span class="use">Three-well toy profile of demo 1; background.</span></li>
      <li id="ref3">A. Barducci, M. Bonomi and M. Parrinello, “Metadynamics,” <i>WIREs Comput. Mol. Sci.</i> <b>1</b>, 826–843 (2011). <a href="https://doi.org/10.1002/wcms.31">doi:10.1002/wcms.31</a> <span class="use">Eqs. 1.2–1.4 (bias and its limit), 1.6 (error), 1.7–1.11 (well-tempered), 1.13 (reweighting).</span></li>
      <li id="ref4">D. Ray and M. Parrinello, “Kinetics from Metadynamics: Principles, Applications, and Outlook,” <i>J. Chem. Theory Comput.</i> <b>19</b>, 5649–5670 (2023). <a href="https://doi.org/10.1021/acs.jctc.3c00660">doi:10.1021/acs.jctc.3c00660</a> <span class="use">Eqs. 11–14 (time rescaling, Poisson statistics, KS test, confidence interval); demo 3.</span></li>
      <li id="ref5">A. Barducci, G. Bussi and M. Parrinello, “Well-tempered metadynamics: a smoothly converging and tunable free-energy method,” <i>Phys. Rev. Lett.</i> <b>100</b>, 020603 (2008). <a href="https://doi.org/10.1103/PhysRevLett.100.020603">doi:10.1103/PhysRevLett.100.020603</a> <span class="use">Well-tempered metadynamics; demo 2.</span></li>
      <li id="ref6">G. Bussi, A. Laio and M. Parrinello, “Equilibrium free energies from nonequilibrium metadynamics,” <i>Phys. Rev. Lett.</i> <b>96</b>, 090601 (2006). <a href="https://doi.org/10.1103/PhysRevLett.96.090601">doi:10.1103/PhysRevLett.96.090601</a> <span class="use">Convergence proof and error scaling for Langevin dynamics.</span></li>
      <li id="ref7">G. M. Torrie and J. P. Valleau, “Nonphysical sampling distributions in Monte Carlo free-energy estimation: umbrella sampling,” <i>J. Comput. Phys.</i> <b>23</b>, 187–199 (1977). <a href="https://doi.org/10.1016/0021-9991(77)90121-8">doi:10.1016/0021-9991(77)90121-8</a> <span class="use">Umbrella sampling and reweighting.</span></li>
      <li id="ref8">H. Grubmüller, “Predicting slow structural transitions in macromolecular systems: conformational flooding,” <i>Phys. Rev. E</i> <b>52</b>, 2893–2906 (1995). <a href="https://doi.org/10.1103/PhysRevE.52.2893">doi:10.1103/PhysRevE.52.2893</a> <span class="use">Flooding: rescaling time with a basin-only bias.</span></li>
      <li id="ref9">P. Tiwary and M. Parrinello, “From metadynamics to dynamics,” <i>Phys. Rev. Lett.</i> <b>111</b>, 230602 (2013). <a href="https://doi.org/10.1103/PhysRevLett.111.230602">doi:10.1103/PhysRevLett.111.230602</a> <span class="use">Infrequent metadynamics.</span></li>
      <li id="ref10">M. Salvalaglio, P. Tiwary and M. Parrinello, “Assessing the reliability of the dynamics reconstructed from metadynamics,” <i>J. Chem. Theory Comput.</i> <b>10</b>, 1420–1425 (2014). <a href="https://doi.org/10.1021/ct500040r">doi:10.1021/ct500040r</a> <span class="use">The KS test against a Poisson model.</span></li>
      <li id="ref11">H. W. Lilliefors, “On the Kolmogorov–Smirnov test for the exponential distribution with mean unknown,” <i>J. Am. Stat. Assoc.</i> <b>64</b>, 387–389 (1969). <a href="https://doi.org/10.1080/01621459.1969.10500983">doi:10.1080/01621459.1969.10500983</a> <span class="use">The stricter test when τ<sub>c</sub> is fitted from the data.</span></li>
      <li id="ref12">K. S. Kaminsky, “Confidence intervals for the exponential scale parameter using optimally selected order statistics,” <i>Technometrics</i> <b>14</b>, 371–383 (1972). <a href="https://doi.org/10.1080/00401706.1972.10488922">doi:10.1080/00401706.1972.10488922</a> <span class="use">The χ² confidence interval of JCTC eq. 14.</span></li>
      <li id="ref13">M. Bonomi, A. Barducci and M. Parrinello, “Reconstructing the equilibrium Boltzmann distribution from well-tempered metadynamics,” <i>J. Comput. Chem.</i> <b>30</b>, 1615–1621 (2009). <a href="https://doi.org/10.1002/jcc.21305">doi:10.1002/jcc.21305</a> <span class="use">Reweighting well-tempered runs (WIREs eqs. 1.12–1.13).</span></li>
    </ol>
    <p class="foot">Software: NumPy and SciPy (simulations, <code>curve_fit</code>, <code>ks_2samp</code>, <code>chi2</code>), MathJax (equations). Every reference above was checked against the reference lists of [3] and [4] or against the papers themselves, and each DOI against the publisher or an indexing record.</p>
  </section>
'''


FILES = [
 ('demos.py', 'The three 1D demonstrations as static figures: hill filling (demo 1), standard vs well-tempered (demo 2) and infrequent-metadynamics kinetics (demo 3). Writes three PNG files and demo_results.json.', 'about 1 minute'),
 ('record.py', 'The same three runs with the same seed (7), recording every hill, trajectory point and walker clock. Writes sim.json, which this page replays. Its results match demos.py exactly.', 'about 1 minute'),
 ('ratestats.py', 'Shared helpers for the kinetics: linear interpolation of the bias grid, the fit of the Poisson CDF, the bootstrap-calibrated KS test and the χ² confidence interval.', None),
 ('surf.py', 'The two 2D model surfaces (ion pair and dipeptide-like map) and the minimax path search used to measure saddle heights. Imported by the next two scripts.', None),
 ('sim2002.py', '2D metadynamics on both models. Each model restarts the seed 2002, so the two runs are independent and reproducible. Writes raw2002.json.', 'about 45 seconds'),
 ('pre2002.py', 'Turns raw2002.json into the per-frame barrier readouts and true-surface grids this page uses. Writes sim2002.json.', 'about 30 seconds'),
 ('validate.py', 'The numerical checks shown in the Checks panels: rate statistics over many independent walkers, free-energy recovery over several seeds, boundary treatment, interpolation accuracy, the periodic seam and grid convergence. Writes validation.json.', 'about 10 minutes'),
]
def source_section():
    items = []
    for fn, what, rt in FILES:
        body = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulations', fn)).read()
        n = body.count('\n') + 1
        items.append(f'<details class="x src"><summary><span class="xt">{fn.split(".")[-1]}</span>{fn}<span class="meta">{n} lines · {('runs in ' + rt) if rt else 'module, not run directly'}</span></summary>'
                     f'<div class="xb"><p>{what}</p><div class="codebar"><button type="button" class="copy">Copy code</button></div>'
                     f'<pre class="code"><code>{html.escape(body.rstrip())}</code></pre></div></details>')
    return f"""
  <section class="refs" id="code">
    <p class="eyebrow">Source code</p>
    <h2>Run it yourself</h2>
    <p class="lede">Every number and curve on this page comes from these seven scripts. They need Python 3 with NumPy, SciPy and Matplotlib.</p>
    <pre class="code"><code>pip install numpy scipy matplotlib
python simulations/demos.py      # static figures for demos 1–3
python simulations/record.py     # data/sim.json: full recording of demos 1–3
python simulations/sim2002.py    # data/raw2002.json: the two 2D models
python simulations/pre2002.py    # data/sim2002.json: readouts for the 2D models
python simulations/validate.py   # data/validation.json: the numerical checks</code></pre>
    <div class="more">{''.join(items)}</div>
  </section>
"""

CSS = '''
.more{display:flex;flex-direction:column;gap:8px}
.more-h{font-size:13px;letter-spacing:.12em;text-transform:uppercase;font-weight:600;color:var(--muted)}
details.x{border:1px solid var(--line);border-radius:10px;background:var(--card)}
details.x>summary{cursor:pointer;list-style:none;padding:12px 16px;font-weight:600;color:var(--ink);display:flex;align-items:center;gap:12px}
details.x>summary::-webkit-details-marker{display:none}
details.x>summary::after{content:"+";margin-left:auto;font-family:var(--mono);color:var(--orange);font-size:18px}
details.x[open]>summary::after{content:"−"}
details.x>summary:focus-visible{outline:2px solid var(--orange);outline-offset:2px;border-radius:10px}
.xt{font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--orange);border:1px solid var(--orange);border-radius:999px;padding:2px 8px;min-width:68px;text-align:center}
.xb{padding:4px 18px 18px;display:flex;flex-direction:column;gap:12px;max-width:92ch;color:var(--body)}
.xb ul{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:10px}
.xb code{font-family:var(--mono);font-size:13.5px}
.xb .m{overflow-x:auto;color:var(--ink);padding:4px 0}
.xb .m svg{font-size:16px;max-width:none}
pre.code{margin:0;background:var(--plot);border:1px solid var(--grid);border-radius:8px;padding:12px 14px;overflow-x:auto;font-size:13px;line-height:1.5;color:var(--ink)}
pre.code code{font-size:13px}
table.pt{border-collapse:collapse;font-size:14.5px;width:100%;max-width:640px}
table.pt th,table.pt td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
table.pt th{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
.cite{font-size:.85em;white-space:nowrap}
.cite a,.refs a{color:var(--blue)}
.refs{display:flex;flex-direction:column;gap:14px;scroll-margin-top:16px}
.refs ol{margin:0;padding-left:28px;display:flex;flex-direction:column;gap:10px;max-width:96ch}
.refs li{scroll-margin-top:24px}
.refs li:target{background:var(--hl);border-radius:6px}
.refs .use{display:block;font-size:14px;color:var(--muted)}
body.rec .more{display:none}
.tw{overflow-x:auto;max-width:100%}
.tw table.pt{max-width:none;min-width:560px}
.tw th{text-transform:none;letter-spacing:0;font-size:12.5px}
.src summary .meta{margin-left:auto;font:400 12.5px var(--mono);color:var(--muted)}
.src summary::after{margin-left:16px !important}
.src pre.code{max-height:560px;overflow:auto}
.codebar{display:flex;justify-content:flex-end}
.codebar .copy{font:600 13px var(--sans);color:var(--ink);background:transparent;border:1px solid var(--line);border-radius:999px;padding:5px 12px;cursor:pointer}
.codebar .copy:focus-visible{outline:2px solid var(--orange);outline-offset:2px}
.refs h2{font-size:clamp(24px,3vw,34px)}
'''

def apply(t):
    for sid, items in P.items():
        items = sorted(items, key=lambda s: 1 if 'Numerical checks' in s else 2 if '>Pitfalls<' in s else 0)
        start = t.index(f'<section class="demo" id="{sid}">')
        end = t.index('</section>', start)
        block = '\n    <div class="more"><p class="more-h">Go deeper</p>' + ''.join(items) + '</div>\n  '
        t = t[:end] + block + t[end:]
    # about panel + references before the footnote
    f = t.index('  <p class="foot">')
    t = t[:f] + '  <div class="more">' + ABOUT + '</div>\n' + REFS + source_section() + t[f:]
    t = t.replace('.foot{font-size:14px;color:var(--muted);max-width:90ch}', '.foot{font-size:14px;color:var(--muted);max-width:90ch}' + CSS)
    return t
