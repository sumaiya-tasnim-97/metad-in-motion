# Metadynamics in motion

Interactive, frame-by-frame replays of small metadynamics simulations. Each animation sits beside the equations that drive it. The term that matters at each moment is highlighted, and expandable panels explain the model, the code, what to expect and the common pitfalls.

**Open `index.html` in any browser.** It is a single self-contained page: all data is embedded, and only the web fonts load from Google Fonts.

## What is inside

The page opens with a short introduction to metadynamics (rare events, collective variables, the history-dependent bias, well-tempered metadynamics and rates), followed by five demonstrations:


| Section | What it shows |
|---|---|
| Demonstration 1 | Gaussian hills fill a three-well profile until F + V<sub>G</sub> is flat |
| Demonstration 2 | Standard vs well-tempered metadynamics: shrinking hills and a converging estimate |
| Demonstration 3 | Infrequent metadynamics: rescaled escape times, a Poisson fit, a KS test and a χ² confidence interval, compared with brute-force MD |
| Laio & Parrinello 2002, test A | A two-coordinate ion-pair model: separation requires the solvent coordinate to move |
| Laio & Parrinello 2002, test B | A periodic dipeptide-like map filling one basin at a time |

Tests A and B are toy models built to have the shape and key energies of the paper's figures. They are not the real NaCl or alanine-dipeptide systems.

## Repository layout

```
index.html          the page (built; open this)
simulations/        Python scripts that produce every number on the page
data/               simulation output that the page replays
site/               page template, explanation panels and build script
movies/             1080p MP4 recordings of the five animations
```

## Reproduce the data

Python 3 with NumPy, SciPy and Matplotlib:

```bash
pip install -r requirements.txt
python simulations/demos.py      # static PNG figures for demos 1–3
python simulations/record.py     # data/sim.json      (~45 s)
python simulations/sim2002.py    # data/raw2002.json  (~45 s)
python simulations/pre2002.py    # data/sim2002.json  (~30 s)
```

All runs are seeded (`default_rng(7)` for demos 1–3 and `default_rng(2002)` for each 2D model), so the output is bit-for-bit identical to the files in `data/`.

## Rebuild the page

The equations are pre-rendered to SVG with MathJax (Node.js), and `site/eqs.json` is included so this step is optional.

```bash
cd site
npm install          # only needed to re-render equations
node eqs.js          # writes eqs.json
python build.py      # writes ../index.html
```

## References

The page lists 13 references with DOIs. The core sources:

- A. Laio and M. Parrinello, *PNAS* **99**, 12562 (2002). doi:10.1073/pnas.202427399
- A. Laio and F. L. Gervasio, *Rep. Prog. Phys.* **71**, 126601 (2008). doi:10.1088/0034-4885/71/12/126601
- A. Barducci, M. Bonomi and M. Parrinello, *WIREs Comput. Mol. Sci.* **1**, 826 (2011). doi:10.1002/wcms.31
- D. Ray and M. Parrinello, *J. Chem. Theory Comput.* **19**, 5649 (2023). doi:10.1021/acs.jctc.3c00660

## Note on figures

The page shows small thumbnails of Figs. 2 and 3 of Laio & Parrinello (2002) for comparison, embedded from `site/figures/`. Check the publisher's reuse terms before making this repository public.

## Contact

For questions or concerns, contact Sumaiya Tasnim at tasnims@duq.edu.
