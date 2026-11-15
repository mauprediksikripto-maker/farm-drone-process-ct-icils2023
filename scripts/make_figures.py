"""Recreate Figures 1-4 of the manuscript from the result tables in ../results.

Usage:  python scripts/make_figures.py   (run from the repository root)
Output: figures/Figure_1.(pdf|png) ... Figure_4.(pdf|png)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#8a8a85', 'axes.labelcolor': '#333',
                     'xtick.color': '#555', 'ytick.color': '#555'})
CT, CIL = '#2a78d6', '#eb6834'          # CT on the second module / CIL (discriminant outcome)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: design
fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.8)
def box(x, y, w, h, t, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8.2, color='#222')
box(0.1, 2.3, 2.9, 1.6, 'Farm Drone module\n(all CT students)\n\nProcess indicators:\nresets, time, rapid responses', '#e3eefb', CT)
box(0.1, 0.2, 2.9, 1.4, 'Farm Drone score\n(9 scored items)\nused as a control only', '#f2f2ef', '#8a8a85')
box(3.45, 3.7, 3.5, 1.0, 'Design: a second CT module is\nrandomly rotated (Automated Bus,\nSuns and Moons, Activity Tracker)', '#ffffff', '#8a8a85')
box(7.3, 2.3, 2.6, 1.6, 'Primary outcome:\nCT score on the other\nmodule (no Farm\nDrone items)', '#e3eefb', CT)
box(7.3, 0.2, 2.6, 1.4, 'Discriminant outcome:\nCIL plausible values', '#fdeee7', CIL)
ax.annotate('', xy=(7.25, 3.1), xytext=(3.05, 3.1), arrowprops=dict(arrowstyle='-|>', color='#444', lw=1.1))
ax.text(5.15, 2.85, 'RQ1: predicts beyond the score?', ha='center', va='top', fontsize=7.8, color='#444')
ax.annotate('', xy=(7.25, 0.9), xytext=(3.05, 2.4), arrowprops=dict(arrowstyle='-|>', color='#999', lw=0.9, ls='--'))
ax.text(5.0, 1.15, 'RQ2: specific to CT?', ha='center', fontsize=7.8, color='#777')
ax.annotate('', xy=(8.6, 3.95), xytext=(7.0, 4.2),
            arrowprops=dict(arrowstyle='-|>', color='#999', lw=0.9, connectionstyle='arc3,rad=-0.2'))
fig.tight_layout(); save(fig, 'Figure_1')

# ---------------------------------------------------------------- Figure 2: stepwise coefficients
r = pd.read_csv(os.path.join(RES, '03_stepwise_models.csv'))
P = [('reset_rate', 'Reset rate'), ('fd_time', 'Farm Drone time'), ('rapid_prop', 'Rapid responses'), ('n_tasks', 'Tasks attempted')]
M = ['M0', 'M1', 'M2', 'M3']
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
for ax, (p, title) in zip(axs, P):
    for oc, col, mk, dx in [('CT_second_module', CT, 'o', -0.09), ('CIL', CIL, 's', 0.09)]:
        d = r[(r.predictor == p) & (r.outcome == oc)].set_index('model').loc[M]
        x = np.arange(4) + dx
        ax.plot(x, d.coef, color=col, lw=1.2, alpha=0.6)
        ax.errorbar(x, d.coef, yerr=1.96 * d.SE, fmt=mk, color=col, ms=4, lw=1.2, capsize=0)
    ax.axhline(0, color='#8a8a85', lw=0.8); ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(M, fontsize=8)
    ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('Coefficient (SD per SD), 95% CI')
axs[0].plot([], [], 'o', color=CT, label='CT on second module'); axs[0].plot([], [], 's', color=CIL, label='CIL')
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, 'Figure_2')

# ---------------------------------------------------------------- Figure 3: exploratory (task group, tertile)
ext = pd.read_csv(os.path.join(RES, '05_exploratory.csv'))
rows = [('b. reset rate by task group', 'reset_basic', 'Basic tasks (F01–F02)'),
        ('b. reset rate by task group', 'reset_building', 'Building tasks (F04, F05, F07)'),
        ('b. reset rate by task group', 'reset_debugging', 'Debugging tasks (F03, F06, F08)'),
        ('c. Farm Drone score tertile: low', 'reset_rate', 'Low Farm Drone score'),
        ('c. Farm Drone score tertile: middle', 'reset_rate', 'Middle Farm Drone score'),
        ('c. Farm Drone score tertile: high', 'reset_rate', 'High Farm Drone score')]
fig, ax = plt.subplots(figsize=(6.3, 3.6))
ypos = [5.4, 4.4, 3.4, 1.6, 0.6, -0.4]
for (an, pr, lab), y in zip(rows, ypos):
    for oc, col, mk, dy in [('CT_second_module', CT, 'o', 0.13), ('CIL', CIL, 's', -0.13)]:
        q = ext[(ext.analysis == an) & (ext.predictor == pr) & (ext.outcome == oc)].iloc[0]
        ax.errorbar(q.coef, y + dy, xerr=1.96 * q.SE, fmt=mk, color=col, ms=5, lw=1.4, capsize=0)
ax.axvline(0, color='#8a8a85', lw=0.8)
ax.set_yticks(ypos); ax.set_yticklabels([x[2] for x in rows])
ax.text(-0.135, 6.05, 'A. By task type', fontweight='bold', fontsize=9)
ax.text(-0.135, 2.25, 'B. By Farm Drone score tertile', fontweight='bold', fontsize=9)
ax.set_xlim(-0.14, 0.11); ax.set_ylim(-0.9, 6.3)
ax.set_xlabel('Coefficient of reset rate (SD of outcome per SD), 95% CI')
ax.plot([], [], 'o', color=CT, label='CT on other module'); ax.plot([], [], 's', color=CIL, label='CIL (discriminant)')
ax.legend(frameon=False, loc='upper right', fontsize=8)
ax.grid(axis='x', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, 'Figure_3')

# ---------------------------------------------------------------- Figure 4: by system
p = pd.read_csv(os.path.join(RES, '04_by_system.csv'))
p['se'] = p.reset_rate / p.reset_rate_t
p = p.sort_values('reset_rate', ascending=False).reset_index(drop=True)
pooled = r[(r.outcome == 'CT_second_module') & (r.predictor == 'reset_rate') & (r.model == 'M2')].iloc[0]
fig, ax = plt.subplots(figsize=(5.2, 6.4))
y = np.arange(len(p)); sig = p.reset_rate_t.abs() > 2
ax.errorbar(p.reset_rate, y, xerr=1.96 * p.se, fmt='none', ecolor=CT, lw=1.2)
ax.scatter(p.reset_rate[sig], y[sig], color=CT, s=22, zorder=3, label='|t| > 2')
ax.scatter(p.reset_rate[~sig], y[~sig], facecolor='white', edgecolor=CT, s=22, zorder=3, label='|t| ≤ 2')
yp = len(p) + 0.8
ax.errorbar(pooled.coef, yp, xerr=1.96 * pooled.SE, fmt='D', color='#333', ms=5, lw=1.4)
ax.set_yticks(list(y) + [yp]); ax.set_yticklabels([f'{c} (n={n})' for c, n in zip(p.system, p.n)] + ['Pooled, country FE'])
ax.axvline(0, color='#8a8a85', lw=0.8); ax.set_xticks([-0.15, -0.10, -0.05, 0.0])
ax.set_xlabel('Coefficient of reset rate on CT, other module\n(SD per SD, model M2, 95% CI)')
ax.legend(frameon=False, loc='lower left', fontsize=8)
ax.grid(axis='x', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, 'Figure_4')
print('Figures written to', OUT)
