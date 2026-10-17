# Per-frame readouts for the 2002 models: 1D barrier from the learned map (A) and the
# C7eq -> saddle barrier via a minimax path search on the periodic learned map (B).
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
sys.path.insert(0, HERE)
import numpy as np, json
from surf import *
d = json.load(open(os.path.join(DATA, 'raw2002.json')))
HA = np.array(d['A']['hills']); HB = np.array(d['B']['hills'])[:700]
TA = d['A']['traj']; TB = d['B']['traj'][:700*150//30]
# ---- A
r = np.linspace(2.2,6,96); y = np.linspace(0,1.2,61); R,Y = np.meshgrid(r,y,indexing='ij')
G = FA(R,Y); G1 = -kT*np.log(np.exp(-G/kT).sum(1)); G1 -= G1[r<3.3].min(); trueA = float(G1[(r>3.5)&(r<4.5)].max())
V = np.zeros_like(R); barA = []; k = 0; FA_frames = list(range(2, 1401, 2))
for kk in FA_frames:
    while k < kk:
        V += 0.3*np.exp(-(R-HA[k,0])**2/(2*.12**2)-(Y-HA[k,1])**2/(2*.04**2)); k += 1
    if HA[:k,0].max() < 4.8: barA.append(None); continue
    E = -V; E -= E[r<3.3].min(); F1 = -kT*np.log(np.exp(-E/kT).sum(1)); F1 -= F1[r<3.3].min()
    barA.append(round(float(F1[(r>3.5)&(r<4.5)].max()),2))
# ---- B
ps = np.linspace(-180,180,73)[:-1]; P,Q = np.meshgrid(ps,ps,indexing='ij')
def w(a): return (a+180)%360-180
B = FB(P,Q); B -= B.min()
mn = lambda E,c: np.unravel_index(np.argmin(np.where(c,E,99)),E.shape)
cA = (abs(P+30)<40)&(abs(Q+63)<40); cC = (P>120)&(abs(Q+75)<30)
ia, ic = mn(B,cA), mn(B,cC); trueB = float(bottleneck(B,ia,ic,True)-B[ic]); trueC7 = float(B[ic])
V = np.zeros_like(P); barB = []; c7B = []
for k in range(len(HB)):
    V += 0.2*np.exp(-(w(P-HB[k,0])**2+w(Q-HB[k,1])**2)/(2*14**2))
    E = -V; E -= E.min(); ia, ic = mn(E,cA), mn(E,cC)
    visitedA = (np.abs(HB[:k+1,0]+30)<35).any() and (np.abs(HB[:k+1,1]+63)<35).any()
    barB.append(round(float(bottleneck(E,ia,ic,True)-E[ic]),2) if visitedA and V[ia]>0.5 else None)
    c7B.append(round(float(E[ic]-E[ia]),2) if visitedA and V[ia]>0.5 else None)
ev = lambda cond: next((i for i,h in enumerate(HB) if cond(h)), None)
out = {'kT': kT,
 'A': {'hills': HA.round(4).tolist(), 'traj': TA, 'bar': barA, 'trueBar': round(trueA,2), 'W': 0.3, 'sr': 0.12, 'sy': 0.04, 'tau': 250, 'trajEvery': 50,
       'diss': next(i for i,h in enumerate(HA) if h[0]>4.8)},
 'B': {'hills': HB.round(2).tolist(), 'traj': TB, 'bar': barB, 'c7': c7B, 'trueBar': round(trueB,2), 'trueC7': round(trueC7,2), 'W': 0.2, 's': 14, 'tau': 150, 'trajEvery': 30,
       'evAR': ev(lambda h: abs(h[0]+30)<35 and abs(h[1]+63)<35), 'evAL': ev(lambda h: 20<h[1]<110)}}

# true surfaces on the display grids (grey overlay and dashed 1D profile)
Gd = G - G.min()
out['A']['trueF'] = np.minimum(Gd, 30).round(2).ravel().tolist(); out['A']['trueF1'] = G1.round(3).tolist()
ax = np.linspace(-180,180,73); P2,Q2 = np.meshgrid(ax,ax,indexing='ij'); B2 = FB(P2,Q2); B2 -= B2.min()
out['B']['trueF'] = B2.round(2).ravel().tolist()
import sys
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'sim2002.json'),'w'), separators=(',',':'))
print('A true', trueA, 'diss', out['A']['diss'], 'bar end', barA[-5:])
print('B true', trueB, trueC7, out['B']['evAR'], out['B']['evAL'], barB[::50], c7B[::50])
