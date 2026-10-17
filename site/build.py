# Builds ../index.html from the page template, the explanation panels and the simulation data.
# Run from this folder after `npm install` and `node eqs.js` (which writes eqs.json).
import os
import panels
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
rd = lambda *p: open(os.path.join(*p)).read()
t = panels.apply(rd(HERE, 'template.html'))
t = (t.replace('/*SIM2*/null', rd(ROOT, 'data', 'sim2002.json'))
      .replace('/*FIGA*/', rd(HERE, 'figures', 'figA.b64')).replace('/*FIGB*/', rd(HERE, 'figures', 'figB.b64'))
      .replace('/*SIM*/null', rd(ROOT, 'data', 'sim.json')).replace('/*EQS*/null', rd(HERE, 'eqs.json')))
i = t.index('<div class="wrap">')
head, body = t[:i], t[i:]
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       + head + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
open(os.path.join(ROOT, 'index.html'), 'w').write(doc)
print('wrote index.html,', len(doc), 'bytes')
