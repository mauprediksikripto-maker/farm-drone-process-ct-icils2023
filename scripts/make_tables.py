"""Recreate the manuscript tables (as CSV) from the result files in ../results.

Usage:  python scripts/make_tables.py   (run from the repository root)
Output: tables/Table_2.csv ... tables/Table_A4.csv
(Table 1 lists variable definitions and has no data file.)
Coefficients carry significance markers (* p < .05, ** p < .01, *** p < .001; |t| >= 1.96, 2.58, 3.29).
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'tables')
os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(RES, f))


def star(t):
    t = abs(t)
    return '***' if t >= 3.29 else '**' if t >= 2.58 else '*' if t >= 1.96 else ''


def cell(q):
    return f'{q.coef:.3f}{star(q.t)} ({q.SE:.3f})'


def ci(q):
    return f'{q.coef:.3f}{star(q.t)} [{q.coef - 1.96 * q.SE:.3f}, {q.coef + 1.96 * q.SE:.3f}]'


LAB = {'n_tasks': 'Tasks attempted', 'fd_time': 'Farm Drone time', 'reset_rate': 'Reset rate',
       'rapid_prop': 'Rapid responses', 'fd_score': 'Farm Drone score', 'second_time': 'Second-module time',
       'fd_time_sq': 'Farm Drone time, squared', 'ct_second': 'CT on second module',
       'female': 'Female', 'SES': 'Socio-economic background (IISB)'}

# Table 2: descriptives and correlations
d = pd.read_csv(os.path.join(RES, '02_descriptives.csv'), index_col=0)
c = pd.read_csv(os.path.join(RES, '02b_correlations.csv'), index_col=0)
V = ['n_tasks', 'fd_time', 'reset_rate', 'rapid_prop', 'fd_score', 'second_time', 'ct_second']
rows = []
for i, v in enumerate(V):
    r = {'Variable': f'{i + 1}. {LAB[v]}', 'n': int(d.loc[v, 'count']), 'M': round(d.loc[v, 'mean'], 3), 'SD': round(d.loc[v, 'std'], 2)}
    for j in range(len(V) - 1):
        r[str(j + 1)] = round(c.loc[v, V[j]], 2) if j < i else ''
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'Table_2.csv'), index=False)

# Table 3: stepwise models
s = rd('03_stepwise_models.csv')
cols = [('CT_second_module', 'M0'), ('CT_second_module', 'M1'), ('CT_second_module', 'M2'), ('CT_second_module', 'M3'), ('CIL', 'M2')]
rows = []
for p in ['n_tasks', 'fd_time', 'reset_rate', 'rapid_prop', 'fd_score', 'second_time', 'fd_time_sq']:
    r = {'Predictor': LAB[p]}
    for oc, m in cols:
        q = s[(s.predictor == p) & (s.outcome == oc) & (s.model == m)]
        r[f'{oc} {m}'] = cell(q.iloc[0]) if len(q) else ''
    rows.append(r)
r2 = {'Predictor': 'R2'}; nn = {'Predictor': 'n'}
for oc, m in cols:
    q = s[(s.outcome == oc) & (s.model == m)].iloc[0]
    r2[f'{oc} {m}'] = f'{q.R2:.3f}'; nn[f'{oc} {m}'] = int(q.n)
rows += [r2, nn]
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'Table_3.csv'), index=False)

# Table 4: exploratory analyses
e = rd('05_exploratory.csv')
spec = [('Reset rate (interaction model)', 'a. M2 + reset x Farm Drone score', 'reset_rate'),
        ('Reset rate x Farm Drone score', 'a. M2 + reset x Farm Drone score', 'reset_x_score'),
        ('Basic tasks', 'b. reset rate by task group', 'reset_basic'),
        ('Building tasks', 'b. reset rate by task group', 'reset_building'),
        ('Debugging tasks', 'b. reset rate by task group', 'reset_debugging'),
        ('Low score tertile', 'c. Farm Drone score tertile: low', 'reset_rate'),
        ('Middle score tertile', 'c. Farm Drone score tertile: middle', 'reset_rate'),
        ('High score tertile', 'c. Farm Drone score tertile: high', 'reset_rate')]
rows = []
for lab, an, pr in spec:
    r = {'Specification': lab}
    for oc in ['CT_second_module', 'CIL']:
        q = e[(e.analysis == an) & (e.predictor == pr) & (e.outcome == oc)].iloc[0]
        r[f'{oc}: beta [95% CI]'] = ci(q)
        if oc == 'CT_second_module':
            r['n'] = int(q.n)
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'Table_4.csv'), index=False)

# Table A.4: by system
p = rd('04_by_system.csv').sort_values('system')
a1 = pd.DataFrame({'System': p.system, 'n': p.n})
for k in ['reset_rate', 'fd_time', 'rapid_prop', 'fd_score']:
    a1[f'{LAB[k]} beta (t)'] = [f'{b:.3f} ({t:.2f})' for b, t in zip(p[k], p[k + '_t'])]
a1.to_csv(os.path.join(OUT, 'Table_A4.csv'), index=False)

# Tables A.1-A.3 (sample, rotation balance, robustness)
rd('06_sample_by_system.csv').to_csv(os.path.join(OUT, 'Table_A1.csv'), index=False)
rd('07_rotation_balance.csv').assign(variable=lambda x: x.variable.map(lambda v: LAB.get(v, v))).to_csv(os.path.join(OUT, 'Table_A2.csv'), index=False)
rb = rd('08_robustness_gender_ses.csv')
rows = []
for pr in ['reset_rate', 'fd_time', 'rapid_prop', 'n_tasks', 'fd_score', 'second_time', 'female', 'SES']:
    if pr not in set(rb.predictor):
        continue
    r = {'Predictor': LAB[pr]}
    for oc in ['CT_second_module', 'CIL']:
        r[oc] = cell(rb[(rb.predictor == pr) & (rb.outcome == oc)].iloc[0])
    rows.append(r)
rows.append({'Predictor': 'R2', **{oc: f"{rb[rb.outcome == oc].R2.iloc[0]:.3f}" for oc in ['CT_second_module', 'CIL']}})
rows.append({'Predictor': 'n', **{oc: int(rb[rb.outcome == oc].n.iloc[0]) for oc in ['CT_second_module', 'CIL']}})
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'Table_A3.csv'), index=False)
print('Tables written to', OUT, sorted(os.listdir(OUT)))
