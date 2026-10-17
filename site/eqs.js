const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor(); RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {InputJax: new TeX({packages: AllPackages}), OutputJax: new SVG({fontCache: 'none'})});
const E = {
 i_rate: String.raw`\tau\;\propto\;e^{\Delta F^{\ddagger}/k_BT},\qquad \frac{\tau(20\,k_BT)}{\tau(0)}\approx e^{20}\approx 5\times10^{8}`,
 i_F: String.raw`F(s)=-k_BT\,\ln P(s),\qquad P(s)=\big\langle\,\delta\big(s-s(\mathbf R)\big)\big\rangle`,
 i_hill: String.raw`\begin{gathered}V_G(s,t)=\sum_{t'=\tau_G,2\tau_G,\dots}^{t'<t}W\exp\!\left(-\frac{\big(s-s(t')\big)^2}{2\,\delta s^2}\right)\\[4pt] V_G(s,t\to\infty)=-F(s)+C\end{gathered}`,
 i_wt: String.raw`\begin{gathered}W=\omega\,\tau_G\,e^{-V(s(t),t)/k_B\Delta T}\\[4pt] V(s,t\to\infty)=-\frac{\Delta T}{T+\Delta T}\,F(s)+C\end{gathered}`,
 i_resc: String.raw`\tau=\sum_i\Delta t\;e^{\beta V(s_i,t_i)}\qquad\Pr(\text{no escape by }t)=e^{-t/\tau}`,
 x_em: String.raw`s_{n+1}=s_n-D\big[F'(s_n)+\partial_sV_G(s_n,t_n)\big]\Delta t+\sqrt{2D\,k_BT\,\Delta t}\;\xi_n,\qquad \xi_n\sim\mathcal N(0,1)`,
 x_F3: String.raw`F(s)=-5\,e^{-(s-2)^2/0.75^2}-4\,e^{-s^2/0.75^2}-7\,e^{-(s+2)^2/0.75^2}`,
 x_dep: String.raw`V_G(s_j)\;\leftarrow\;V_G(s_j)+w\,e^{-(s_j-s_n)^2/2\delta s^2}\qquad\text{when } n \bmod 100=0`,
 x_grad: String.raw`\partial_sV_G(s_n)\approx\frac{V_G(s_{j+1})-V_G(s_{j-1})}{2\,\Delta g},\qquad \Delta g=0.01`,
 x_refl: String.raw`s>4\ \Rightarrow\ s\to 8-s,\qquad s<-4\ \Rightarrow\ s\to -8-s`,
 x_omega: String.raw`\omega=\frac{w}{\tau_G}=\frac{0.15\,k_BT}{0.1}=1.5\ k_BT\ \text{per unit time},\qquad \varepsilon\propto\sqrt{\frac{\omega}{D\beta}}`,
 x_fill: String.raw`N_{\rm hills}\approx\frac{\displaystyle\int_{\rm well}\big[F_{\ddagger}-F(s)\big]\,ds}{w\sqrt{2\pi}\,\delta s}=\frac{5.43}{0.15\times\sqrt{2\pi}\times 0.25}\approx 58`,
 x_wtW: String.raw`W_n=w\,\exp\!\Big(-\frac{V(s_n,t_n)}{k_B\Delta T}\Big),\qquad \Delta T=5\,k_BT\ \ (\texttt{dT=5.0})`,
 x_wtF: String.raw`\hat F_{\rm WT}(s)=-\frac{T+\Delta T}{\Delta T}\,V(s,t)=-\Big(1+\frac{1}{\Delta T}\Big)V=-1.2\,V`,
 x_rms: String.raw`d_j=\hat F(s_j)-F(s_j),\qquad \varepsilon=\sqrt{\frac1M\sum_{|s_j|<3.2}\big(d_j-\bar d\,\big)^2}`,
 x_P11: String.raw`P(s)\propto e^{-F(s)/k_B(T+\Delta T)}=e^{-F(s)/6k_BT}`,
 x_F2: String.raw`F(s)=h\,(s^2-1)^2,\qquad F'(s)=4hs\,(s^2-1),\qquad h=8`,
 x_kram: String.raw`\tau_{\rm Kramers}\approx\frac{2\pi}{D\sqrt{F''(-1)\,|F''(0)|}}\,e^{\beta\Delta F}=\frac{2\pi}{\sqrt{64\times 32}}\,e^{8}\approx 414`,
 x_resc: String.raw`\tau_{\rm run}=\sum_{i=1}^{n_{\rm steps}}\Delta t\;e^{\beta V(s_i,t_i)},\qquad t_b=n_{\rm steps}\,\Delta t,\qquad \alpha=\frac{\tau_{\rm run}}{t_b}`,
 x_fit: String.raw`\tau_c=\arg\min_{\tau}\sum_{k=1}^{n}\Big[\tfrac{k}{n}-\big(1-e^{-t_{(k)}/\tau}\big)\Big]^2`,
 x_ks: String.raw`D_{nm}=\sup_t\big|\hat F_n(t)-\hat G_m(t)\big|,\qquad p=\Pr\big(D\ge D_{nm}\mid\text{same distribution}\big)`,
 x_cinum: String.raw`\sum_i\tau_i=23\,695,\quad n=40:\qquad \left[\frac{47\,390}{106.6},\ \frac{47\,390}{57.2}\right]=[444,\ 829]`,
 x_FA: String.raw`F(r,y)=U(r)+45\,\big[y-y_p(r)\big]^2,\qquad y_p(r)=0.50+\frac{0.48}{1+e^{-(r-4)/0.35}},\qquad v=100\,y`,
 x_U: String.raw`U(r)=2.15\,e^{-\left(\frac{r-4}{0.42}\right)^2}-1.2\,e^{-\left(\frac{r-2.8}{0.32}\right)^2}-1.6\,e^{-\left(\frac{r-5.4}{0.55}\right)^2}+\text{walls}`,
 x_lang2: String.raw`x_{n+1}=x_n-D_x\,\partial_x(F+V_G)\,\Delta t+\sqrt{2D_xk_BT\Delta t}\,\xi^x_n,\qquad D_r=1,\ D_y=0.12`,
 x_marg: String.raw`F_1(r_i)=-k_BT\ln\sum_j e^{-\beta F_2(r_i,v_j)}+\text{const}`,
 x_FB: String.raw`F(\Psi,\Phi)=-\sum_{k}A_k\,e^{\kappa^{(k)}_\Psi[\cos(\Psi-\Psi_k)-1]+\kappa^{(k)}_\Phi[\cos(\Phi-\Phi_k)-1]}+3\,e^{8(\cos\Phi-1)}+3\,e^{8[\cos(\Phi-130^\circ)-1]}`,
 x_bott: String.raw`\Delta F^{\ddagger}=\min_{\gamma:\,\mathrm{C7eq}\to\alpha_r}\ \max_{x\in\gamma}\hat F(x)\;-\;\hat F(\mathrm{C7eq})`,
 a_hill: String.raw`V_G(r,v,t)=\sum_{t'<t}\class{t-w}{W}\exp\!\left(-\frac{\big(r-\class{t-sp}{r(t')}\big)^2}{2\sigma_r^2}-\frac{\big(v-\class{t-sp2}{v(t')}\big)^2}{2\sigma_v^2}\right)`,
 a_est: String.raw`F_2(r,v)\approx\class{t-neg}{-V_G(r,v)}+C`,
 a_marg: String.raw`F_1(r)=-k_BT\,\ln\!\class{t-int}{\int\!dv\;e^{-\beta F_2(r,v)}}`,
 b_hill: String.raw`V_G(\Psi,\Phi)=\sum_{t'<t}\class{t-w}{W}\exp\!\left(-\frac{\class{t-d}{d}(\Psi,\Psi')^2+\class{t-d2}{d}(\Phi,\Phi')^2}{2\sigma^2}\right)`,
 b_wrap: String.raw`\class{t-d}{d(a,b)=\big[(a-b+180^\circ)\bmod 360^\circ\big]-180^\circ}`,
 b_bar: String.raw`\Delta F^{\ddagger}=\class{t-s}{F(\text{saddle})}-\class{t-c}{F(\mathrm{C7eq})}\qquad F\approx\class{t-neg}{-V_G}`,
 d1_hill: String.raw`V_G(s,t)=\sum_{t'=\tau_G,2\tau_G,\dots}^{t'<t}\class{t-w}{w}\;\class{t-g}{\exp\!\left(-\frac{\big(s-\class{t-sp}{s(t')}\big)^2}{2\,\delta s^2}\right)}`,
 d1_lang: String.raw`\dot s=-D\,\partial_s\big[\,\class{t-F}{F(s)}+\class{t-V}{V_G(s,t)}\,\big]+\class{t-noise}{\sqrt{2D\,k_BT}\;\eta(t)}`,
 d1_lim: String.raw`\lim_{t\to\infty}V_G(s,t)=\class{t-neg}{-F(s)}+C`,
 d2_W: String.raw`W=\class{t-w}{w}\;\class{t-damp}{\exp\!\left(-\frac{V\big(s(t),t\big)}{k_B\Delta T}\right)}`,
 d2_lim: String.raw`V(s,t\to\infty)=\class{t-frac}{-\frac{\Delta T}{T+\Delta T}}\,F(s)`,
 d2_est: String.raw`\class{t-std}{\hat F_{\rm std}=-V_G}\qquad\class{t-wt}{\hat F_{\rm WT}=-\frac{T+\Delta T}{\Delta T}\,V}`,
 d2_err: String.raw`\varepsilon(t)=\sqrt{\Big\langle\big(\hat F(s)-F(s)-c\big)^2\Big\rangle_{|s|<3.2}}`,
 d3_rate: String.raw`\class{t-k}{\Pr(\text{still in A at }t)=e^{-t/\tau}}`,
 d3_clock: String.raw`\tau=\sum_{i}\class{t-dt}{\Delta t}\;\class{t-exp}{e^{\beta V(s_i,t_i)}}\qquad \class{t-alpha}{\alpha=\frac{\tau}{t_b}}`,
 d3_cdf: String.raw`P_{n\ge1}(t)=\class{t-cdf}{1-e^{-t/\tau_c}}`,
 d3_ci: String.raw`\class{t-ks}{\text{KS: }p>0.05}\qquad \class{t-ci}{\left[\frac{2\sum_i\tau_i}{\chi^2_{2n}(0.975)},\ \frac{2\sum_i\tau_i}{\chi^2_{2n}(0.025)}\right]}`,
};
const out = {};
for (const [k, tex] of Object.entries(E)) {
  const node = doc.convert(tex, {display: true});
  let s = adaptor.innerHTML(node);
  s = s.replace(/<svg /, `<svg aria-label="${k}" `);
  out[k] = s;
}
require('fs').writeFileSync('eqs.json', JSON.stringify(out));
console.log(Object.keys(out).length, JSON.stringify(out).length);
console.log(out.d1_lim.slice(0, 600));
