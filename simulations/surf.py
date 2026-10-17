import numpy as np
kT = 0.596  # kcal/mol at 300 K
# ---- A: ion-pair model. x = r (Angstrom), y = solvent coordinate (scaled V_Cl / 100 kcal/mol)
def U(r):  # 1D profile along the channel: C at 2.8 (0), S at 4.0 (3.3), D at 5.4 (-0.3)
    return (2.15*np.exp(-((r-4.0)/0.42)**2) - 1.2*np.exp(-((r-2.8)/0.32)**2) - 1.6*np.exp(-((r-5.4)/0.55)**2)
            + 6.0*np.clip(2.45-r,0,None)**2*10 + 1.2*np.clip(r-5.9,0,None)**2*10)
def ypath(r): return 0.50 + 0.48/(1+np.exp(-(r-4.0)/0.35))
def FA(r, y): return U(r) + 45*(y-ypath(r))**2
# ---- B: dipeptide-like periodic map, angles in degrees (psi, phi)
WELLS = [  # psi, phi, depth, kpsi, kphi
  (-30, -63, 4.6, 2.05, 3.5),    # alpha_r
  (165, -75, 3.15, 2.05, 3.5),    # C7eq
  (165, -150, 2.6, 4.0, 4.0),   # C5
  (40, 60, 1.6, 3.0, 3.0),      # alpha_L (phi > 0, high)
]
def FB(psi, phi):
    p, f = np.radians(psi), np.radians(phi); out = 0
    for (a, b, d, kp, kf) in WELLS:
        out = out - d*np.exp(kp*(np.cos(p-np.radians(a))-1) + kf*(np.cos(f-np.radians(b))-1))
    out = out + 3.0*np.exp(8*(np.cos(f)-1)) + 3.0*np.exp(8*(np.cos(f-np.radians(130))-1))
    return out
def bottleneck(G, a, b, periodic):
    """minimax path height between grid cells a and b (saddle height)"""
    import heapq
    n, m = G.shape; best = np.full(G.shape, np.inf); best[a] = G[a]; h = [(G[a], a)]
    while h:
        v, (i, j) = heapq.heappop(h)
        if (i, j) == b: return v
        if v > best[i, j]: continue
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii, jj = i+di, j+dj
            if periodic: ii %= n; jj %= m
            elif not (0 <= ii < n and 0 <= jj < m): continue
            nv = max(v, G[ii, jj])
            if nv < best[ii, jj]: best[ii, jj] = nv; heapq.heappush(h, (nv, (ii, jj)))
