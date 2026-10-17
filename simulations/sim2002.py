# 2D metadynamics on the two toy models shaped like Laio & Parrinello (2002), Figs. 2 and 3.
# Overdamped Langevin dynamics, hills kept as a list, bias force summed analytically.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
import numpy as np, json, sys
from surf import *
rng = np.random.default_rng(2002)   # reset per model in __main__
def grad(f, x, y, h=1e-4): return ((f(x+h,y)-f(x-h,y))/(2*h), (f(x,y+h)-f(x,y-h))/(2*h))
# ===== A: ion pair =====
def runA(nhill=1400, W=0.3, sr=0.12, sy=0.04, tau=250, dt=4e-4, Dr=1.0, Dy=0.12):
    x, y = 2.8, 0.5; H = []; traj = []
    hc = np.zeros((0, 2))
    for i in range(nhill*tau):
        if i % tau == 0:
            H.append((round(x,4), round(y,4))); hc = np.array(H)
        if len(hc):
            g = W*np.exp(-(x-hc[:,0])**2/(2*sr*sr) - (y-hc[:,1])**2/(2*sy*sy))
            bx = -(g*(x-hc[:,0])/sr**2).sum(); by = -(g*(y-hc[:,1])/sy**2).sum()
        fx, fy = grad(FA, x, y)
        x += -Dr*(fx+bx)*dt + np.sqrt(2*Dr*kT*dt)*rng.normal()
        y += -Dy*(fy+by)*dt + np.sqrt(2*Dy*kT*dt)*rng.normal()
        if x < 2.2: x = 4.4-x
        if x > 6.0: x = 12-x
        y = min(max(y, 0.0), 1.2)
        if i % 50 == 0: traj.append((round(x,3), round(y,3)))
    return H, traj
def wrap(a): return (a+np.pi) % (2*np.pi) - np.pi
def runB(nhill=1400, W=0.2, s=np.radians(14), tau=150, dt=1e-3, D=0.9):
    p, f = np.radians(165), np.radians(-75); H = []; traj = []; hc = np.zeros((0,2))
    for i in range(nhill*tau):
        if i % tau == 0:
            H.append((round(float(np.degrees(p)),2), round(float(np.degrees(f)),2))); hc = np.radians(np.array(H))
        dp = wrap(p-hc[:,0]); df = wrap(f-hc[:,1])
        g = W*np.exp(-(dp**2+df**2)/(2*s*s)); bp = -(g*dp).sum()/s**2; bf = -(g*df).sum()/s**2
        gp, gf = grad(lambda a,b: FB(np.degrees(a), np.degrees(b)), p, f)
        p = wrap(p - D*(gp+bp)*dt + np.sqrt(2*D*kT*dt)*rng.normal())
        f = wrap(f - D*(gf+bf)*dt + np.sqrt(2*D*kT*dt)*rng.normal())
        if i % 30 == 0: traj.append((round(float(np.degrees(p)),1), round(float(np.degrees(f)),1)))
    return H, traj
if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'raw2002.json')
    rng = np.random.default_rng(2002)          # model A: its own seeded stream
    HA, TA = runA()
    print('A: first hill with r > 4.8 is', next(i for i, (x, y) in enumerate(HA) if x > 4.8))
    rng = np.random.default_rng(2002)          # model B: restart the same seed
    HB, TB = runB()
    print('B: first visit to alpha_r at hill', next(i for i, (a, b) in enumerate(HB) if abs(a+30) < 35 and abs(b+63) < 35))
    json.dump({'A': {'hills': HA, 'traj': TA}, 'B': {'hills': HB, 'traj': TB}}, open(out, 'w'))
