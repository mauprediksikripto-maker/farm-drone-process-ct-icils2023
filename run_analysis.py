"""Script version of farm_drone_analysis.ipynb (identical code).
Usage: ICILS_DIR=/path python run_analysis.py
"""

# ===========================================================================
# SETUP
# ===========================================================================
# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
USE_DRIVE  = True                        # Colab: cache data and results on Google Drive
DRIVE_ROOT = '/content/drive/MyDrive'
CACHE_NAME = 'ICILS2023_farm_drone'      # folder created on Drive (or locally)

NT_FRAC = 0.10    # rapid-response threshold = 10% of the median task time ...
NT_MAX  = 10      # ... capped at 10 seconds (normative-threshold approach)
WINSOR  = 0.99    # reset rates are winsorised at this quantile

import os, sys, subprocess, glob, re, zipfile, shutil, gc, json
IN_COLAB = 'google.colab' in sys.modules
for pkg in ['pyreadstat', 'pyarrow']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
import numpy as np, pandas as pd, pyreadstat

if IN_COLAB and USE_DRIVE and not os.path.isdir(DRIVE_ROOT):
    from google.colab import drive
    drive.mount('/content/drive')
if IN_COLAB:
    BASE_DIR = os.path.join(DRIVE_ROOT, CACHE_NAME) if USE_DRIVE else os.path.join('/content', CACHE_NAME)
else:
    BASE_DIR = os.path.join(os.environ.get('ICILS_DIR', '.'), CACHE_NAME)
RAW_DIR  = os.path.join(BASE_DIR, 'raw_bsg')      # student files BSG*.sav
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoint')   # per-country parquet + model checkpoints
OUT_DIR  = os.path.join(BASE_DIR, 'results')      # result tables (CSV) and summary
for d in (RAW_DIR, CKPT_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

def is_bsg(name):
    b = os.path.basename(name).upper()
    return b.startswith('BSG') and b.endswith('.SAV')

def extract_bsg_from_zip(zpath):
    """Extract only the student files (BSG*.sav) from the IEA SPSS zip, including nested zips."""
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.infolist():
            if is_bsg(m.filename):
                target = os.path.join(RAW_DIR, os.path.basename(m.filename))
                if not os.path.exists(target):
                    with zf.open(m) as s, open(target, 'wb') as d:
                        shutil.copyfileobj(s, d, 16 * 1024 * 1024)
                n += 1
            elif m.filename.upper().endswith('.ZIP'):
                tmp = os.path.join('/tmp', os.path.basename(m.filename))
                with zf.open(m) as s, open(tmp, 'wb') as d:
                    shutil.copyfileobj(s, d, 16 * 1024 * 1024)
                n += extract_bsg_from_zip(tmp)
                os.remove(tmp)
    return n

# Local zips placed in RAW_DIR are unpacked too
for z in glob.glob(os.path.join(RAW_DIR, '*.zip')):
    print(z, '->', extract_bsg_from_zip(z), 'BSG files')

files = sorted(f for f in glob.glob(os.path.join(RAW_DIR, '*')) if is_bsg(f))
if not files and IN_COLAB:
    from google.colab import files as colab_files
    print('No student files cached yet. Upload the ICILS 2023 SPSS zip (or BSG*.sav files).')
    uploaded = colab_files.upload()
    for name in uploaded:
        if name.upper().endswith('.ZIP'):
            print(name, '->', extract_bsg_from_zip(name), 'BSG files')
        elif is_bsg(name):
            shutil.copy(name, os.path.join(RAW_DIR, os.path.basename(name)))
        os.remove(name)
    del uploaded; gc.collect()
    files = sorted(f for f in glob.glob(os.path.join(RAW_DIR, '*')) if is_bsg(f))
if not files:
    raise FileNotFoundError(f'No BSG*.sav files found. Put the ICILS 2023 student files (or the SPSS zip) in {RAW_DIR}')
print(f'{len(files)} student files. Working folder: {BASE_DIR}')

# ===========================================================================
# VARMAP
# ===========================================================================
# CT variables follow the pattern T{cycle}{module}{task}{code}_{suffix}:
#   module F = Farm Drone, A = Automated Bus, S = Suns and Moons, T = Activity Tracker
#   suffix _T = task time (s), _S = scored item, _P = process/product variable
# Only metadata are read here. The role of every variable is written to 01_variable_map.csv.
MAP_PATH = os.path.join(OUT_DIR, '01_variable_map.csv')
CT_MOD = {'F': 'Farm Drone', 'A': 'Automated Bus', 'S': 'Suns and Moons', 'T': 'Activity Tracker'}
OTHER_MODULES = ['Automated Bus', 'Suns and Moons', 'Activity Tracker']
pat = re.compile(r'^T(\d)([A-Z])(\d{2})([A-Z]*)(?:_([A-Z]))?$')

if os.path.exists(MAP_PATH):
    vmap = pd.read_csv(MAP_PATH, keep_default_na=False)
else:
    rows = {}
    for f in files:
        _, meta = pyreadstat.read_sav(f, metadataonly=True)
        for v, lab in meta.column_names_to_labels.items():
            m = pat.match(v.upper())
            if m and m.group(2) in CT_MOD:
                rows[v] = {'var': v, 'label': '' if lab is None else str(lab), 'module': CT_MOD[m.group(2)],
                           'task': int(m.group(3)), 'code': m.group(4), 'suffix': m.group(5) or ''}
        for c in meta.column_names:
            if re.match(r'^PV\dC(T|IL)$', c.upper()):
                rows[c] = {'var': c, 'label': 'plausible value', 'module': 'PV', 'task': 0, 'code': '', 'suffix': ''}
        del meta
    vmap = pd.DataFrame(rows.values())

    def role(r):
        lab = r['label'].lower()
        if r['module'] == 'PV':
            return 'plausible value'
        if r['suffix'] == 'T':
            return 'process: time'
        if r['suffix'] == 'S':
            return 'score'
        if r['suffix'] == 'P' and 'reset' in lab:
            return 'process: reset'
        if r['suffix'] == 'P':
            return 'product/other (not used)'
        return 'score'            # items without a suffix are scored items
    vmap['role'] = vmap.apply(role, axis=1)
    vmap.to_csv(MAP_PATH, index=False)

print(vmap.groupby(['module', 'role']).size().rename('n').to_string())
sel = lambda mod, rl: sorted(vmap[(vmap.module == mod) & (vmap.role == rl)]['var'])
FD_TIME     = sel('Farm Drone', 'process: time')
FD_RESET    = sel('Farm Drone', 'process: reset')
FD_SCORE    = sel('Farm Drone', 'score')
OTHER_SCORE = {m: sel(m, 'score') for m in OTHER_MODULES}
OTHER_TIME  = {m: sel(m, 'process: time') for m in OTHER_MODULES}
print('\nFarm Drone: time vars', len(FD_TIME), '| reset vars', len(FD_RESET), '| scored items', len(FD_SCORE))
print('Scored items, other modules:', {k: len(v) for k, v in OTHER_SCORE.items()})
if not any(OTHER_SCORE.values()):
    raise RuntimeError('Scored items of the other CT modules were not found; check 01_variable_map.csv.')

# ===========================================================================
# LOAD
# ===========================================================================
# Each country file is read once (only the needed columns), converted to float32,
# restricted to students in the CT sample, and saved as parquet. Reruns skip finished files.
BASE   = ['IDCNTRY', 'CNTRY', 'IDSCHOOL', 'IDSTUD', 'TOTWGTS', 'JKZONES', 'JKREPS', 'S_SEX', 'S_GENDER']
PV_CT  = [f'PV{i}CT' for i in range(1, 6)]
PV_CIL = [f'PV{i}CIL' for i in range(1, 6)]
NEED = (BASE + PV_CT + PV_CIL + FD_TIME + FD_RESET + FD_SCORE
        + [v for vs in OTHER_SCORE.values() for v in vs] + [v for vs in OTHER_TIME.values() for v in vs])

def shrink(df):
    for c in df.columns:
        df[c] = df[c].astype('category') if c == 'CNTRY' else pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df

done = skipped = 0
for i, f in enumerate(files, 1):
    ck = os.path.join(CKPT_DIR, os.path.basename(f).rsplit('.', 1)[0] + '.parquet')
    if os.path.exists(ck) or os.path.exists(ck + '.noct'):
        skipped += 1
        continue
    _, meta = pyreadstat.read_sav(f, metadataonly=True)
    cols = set(meta.column_names); del meta
    if not set(PV_CT) <= cols:                       # country without the CT option
        open(ck + '.noct', 'w').close()
        continue
    df, _ = pyreadstat.read_sav(f, usecols=[c for c in NEED if c in cols])
    df = shrink(df)
    df = df[df['PV1CT'].notna()]
    df.to_parquet(ck + '.tmp', index=False); os.replace(ck + '.tmp', ck)
    done += 1
    print(f'[{i}/{len(files)}] {os.path.basename(f)}: {len(df)} students')
    del df; gc.collect()
print(f'Processed: {done} | from checkpoint: {skipped}')

# ===========================================================================
# INDICATORS
# ===========================================================================
# Task groups follow the ICILS item labels of the Farm Drone module
TASK_GROUP = {'F01': 'reset_basic', 'F02': 'reset_basic',
              'F03': 'reset_debugging', 'F06': 'reset_debugging', 'F08': 'reset_debugging',
              'F04': 'reset_building', 'F05': 'reset_building', 'F07': 'reset_building'}
parts = sorted(glob.glob(os.path.join(CKPT_DIR, '*.parquet')))
data = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
cid = 'CNTRY' if 'CNTRY' in data else 'IDCNTRY'

# --- Socio-economic background: the international index is read from the SPSS files ---
_, meta0 = pyreadstat.read_sav(files[0], metadataonly=True)
lab0 = dict(zip(meta0.column_names, meta0.column_labels))
cand = [(c, l) for c, l in lab0.items() if l and ('socio' in l.lower() or 'nisb' in c.lower())]
pref = [c for c, l in cand if 'international' in (l or '').lower()] or [c for c, l in cand]
SESVAR = pref[0] if pref else None
print('SES candidates:', cand, '| used:', SESVAR)
if SESVAR:
    ses_parts = []
    for f in files:
        if SESVAR in pyreadstat.read_sav(f, metadataonly=True)[1].column_names:
            d_, _ = pyreadstat.read_sav(f, usecols=['IDCNTRY', 'IDSTUD', SESVAR])
            ses_parts.append(d_)
    ses = pd.concat(ses_parts, ignore_index=True).rename(columns={SESVAR: 'SES'})
    special = set(float(k) for k in meta0.variable_value_labels.get(SESVAR, {}).keys())
    ses.loc[ses['SES'].isin(special), 'SES'] = np.nan      # labelled missing codes
    ses.loc[ses['SES'].abs() > 10, 'SES'] = np.nan         # unlabelled codes outside a standardised range
    ses['SES'] = ses['SES'].astype('float32')
    print('SES after cleaning:'); print(ses['SES'].describe().round(2).to_string())
    data = data.merge(ses, on=['IDCNTRY', 'IDSTUD'], how='left')
    del ses, ses_parts; gc.collect()

# --- Gender: the code for girls is taken from the SPSS value labels ---
gvar = 'S_GENDER' if 'S_GENDER' in data else ('S_SEX' if 'S_SEX' in data else None)
if gvar:
    labs = meta0.variable_value_labels.get(gvar, {})
    fem = [k for k, v in labs.items() if any(s in str(v).lower() for s in ['girl', 'female'])]
    print('Gender labels', gvar, labs)
    if fem:
        data['female'] = np.where(data[gvar].isna(), np.nan, (data[gvar] == fem[0]).astype(float))

# --- Farm Drone process indicators (attempted tasks = recorded time > 0) ---
tcols = [c for c in FD_TIME if c in data]
T = data[tcols].to_numpy('float64')
attempted = np.where(np.isnan(T), False, T > 0)
Tm = np.where(attempted, T, np.nan)
thr = np.minimum(NT_MAX, NT_FRAC * np.nanmedian(Tm, axis=0))         # task-specific rapid thresholds
print('Rapid-response thresholds (s):', dict(zip(tcols, np.round(thr, 1))))
data['n_tasks'] = attempted.sum(1)
nz = data['n_tasks'].replace(0, np.nan).to_numpy('float64')
with np.errstate(invalid='ignore', divide='ignore'):
    data['fd_time'] = np.where(attempted, np.log1p(np.nan_to_num(T)), 0.0).sum(1) / nz
    data['rapid_prop'] = np.where(attempted, T < thr[None, :], False).sum(1) / nz
    rcols = [c for c in FD_RESET if c in data]
    R = data[rcols].to_numpy('float64')
    tkey = {c[2:5]: j for j, c in enumerate(tcols)}   # match on 'F04' (reset T3F04BA_P <-> time T2F04_T)
    rs_sum = np.zeros(len(data)); tm_sum = np.zeros(len(data))
    grp = {g: [np.zeros(len(data)), np.zeros(len(data))] for g in TASK_GROUP.values()}
    for j, rc in enumerate(rcols):
        if rc[2:5] not in tkey:
            continue
        t = T[:, tkey[rc[2:5]]]; r = R[:, j]
        ok = ~np.isnan(t) & ~np.isnan(r) & (t > 0)
        rs_sum[ok] += r[ok]; tm_sum[ok] += t[ok] / 60
        g = TASK_GROUP.get(rc[2:5])
        if g:
            grp[g][0][ok] += r[ok]; grp[g][1][ok] += t[ok] / 60
    data['reset_rate'] = np.where(tm_sum > 0, rs_sum / np.where(tm_sum > 0, tm_sum, 1), np.nan)   # resets per minute
    for g, (rs, tm) in grp.items():
        data[g] = np.where(tm > 0, rs / np.where(tm > 0, tm, 1), np.nan)
del T, Tm, R, grp; gc.collect()

def prop_score(vs):
    """Mean proportion of the maximum across items; second value flags students who received the items."""
    block = data[[v for v in vs if v in data]].astype('float64')
    mx = block.max().replace(0, np.nan)
    return (block / mx).mean(axis=1), block.notna().any(axis=1)

# Control 1: Farm Drone score (a control only, because the outcome comes from a different module)
data['fd_score'] = prop_score(FD_SCORE)[0] if FD_SCORE else np.nan

# Outcome (CT on the second module, z within module) and control 2 (time on that module = effort)
oc, lt = [], []
data['second_module'] = pd.Series(np.nan, index=data.index, dtype=object)
for mod in OTHER_MODULES:
    vs = [v for v in OTHER_SCORE.get(mod, []) if v in data]
    if not vs:
        continue
    prop, took = prop_score(vs)
    oc.append(((prop - prop[took].mean()) / prop[took].std()).where(took).rename(mod))
    data.loc[took & data['second_module'].isna(), 'second_module'] = mod
    ts = [v for v in OTHER_TIME.get(mod, []) if v in data]
    if ts:
        TT = data[ts].to_numpy('float64'); at = np.where(np.isnan(TT), False, TT > 0)
        cnt = at.sum(1).astype('float64'); cnt[cnt == 0] = np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            lw = np.where(at, np.log1p(np.nan_to_num(TT)), 0.0).sum(1) / cnt
        lt.append(pd.Series(lw, index=data.index).where(took).rename(mod))
        del TT, at
data['ct_second'] = pd.concat(oc, axis=1).mean(axis=1) if oc else np.nan
data['second_time'] = pd.concat(lt, axis=1).mean(axis=1) if lt else np.nan

IND = ['n_tasks', 'fd_time', 'reset_rate', 'rapid_prop']          # process indicators
RG = sorted(set(TASK_GROUP.values()))                               # reset rate by task group
keep = ([cid, 'TOTWGTS', 'JKZONES', 'JKREPS', 'IDSCHOOL', 'ct_second', 'fd_score', 'second_time', 'second_module']
        + [c for c in ['female', 'SES'] if c in data] + IND + RG + [c for c in PV_CT + PV_CIL if c in data])
analytic = data.loc[data['n_tasks'] > 0, [c for c in keep if c in data]].reset_index(drop=True)
del data; gc.collect()

cap = analytic['reset_rate'].quantile(WINSOR)
analytic['reset_rate'] = analytic['reset_rate'].clip(upper=cap)
for g in RG:
    analytic[g] = analytic[g].clip(upper=analytic[g].quantile(WINSOR))
z = lambda s: (s - s.mean()) / s.std()
analytic['reset_x_score'] = z(analytic['reset_rate']) * z(analytic['fd_score'])
analytic['fd_time_sq'] = (analytic['fd_time'] - analytic['fd_time'].mean()) ** 2

# CIL plausible values in SD units (weighted SD of PV1CIL in the analytic sample)
CIL_SD = []
if 'PV1CIL' in analytic:
    ok = analytic['PV1CIL'].notna()
    w = analytic.loc[ok, 'TOTWGTS']; x = analytic.loc[ok, 'PV1CIL']
    m1 = np.average(x, weights=w); sd = np.sqrt(np.average((x - m1) ** 2, weights=w))
    for c in PV_CIL:
        if c in analytic:
            analytic[c + '_sd'] = analytic[c] / sd; CIL_SD.append(c + '_sd')

print(f'Farm Drone students: {len(analytic)} | with a second CT module: {int(analytic.ct_second.notna().sum())} '
      f'| reset winsorising cap: {cap:.2f}/min')
VARS = IND + RG + ['fd_score', 'second_time', 'ct_second']
desc = analytic[VARS].describe().T
desc.to_csv(os.path.join(OUT_DIR, '02_descriptives.csv'))
cor = analytic[VARS].corr().round(2)
cor.to_csv(os.path.join(OUT_DIR, '02b_correlations.csv'))
desc.round(3)

# ===========================================================================
# MODELS
# ===========================================================================
def wls(Xv, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(Xv * sw[:, None], y * sw, rcond=None)[0]

def fit(df, preds, ys, label):
    """Weighted least squares with country fixed effects, standardised predictors,
    JK2 jackknife standard errors (ICILS zones/replicates) and Rubin's rules over plausible values.
    Estimates are checkpointed in CKPT_DIR so an interrupted run resumes."""
    ck = os.path.join(CKPT_DIR, f'fit_{label}.json')
    Z = df[preds].astype('float64'); Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)
    fe = pd.get_dummies(df[cid].astype(str), prefix='C', drop_first=True, dtype='float64')
    X = pd.concat([pd.Series(1.0, index=df.index, name='Intercept'), Z, fe], axis=1)
    names = list(X.columns); Xv = X.to_numpy()
    w0 = df['TOTWGTS'].to_numpy('float64')
    zones = df['JKZONES'].to_numpy(); reps = df['JKREPS'].to_numpy()
    zid = np.unique(zones[~np.isnan(zones)])
    fp = f'{len(df)}|{names}|{ys}'
    store = json.load(open(ck)) if os.path.exists(ck) else {}
    if store.get('_fp') != fp:
        store = {'_fp': fp}
    for y_name in ys:
        if y_name in store:
            continue
        y = df[y_name].to_numpy('float64')
        b = wls(Xv, y, w0); var = np.zeros_like(b)
        for h in zid:                                   # JK2: double one PSU, drop the other
            w = w0.copy(); inz = zones == h
            w[inz & (reps == 1)] *= 2; w[inz & (reps != 1)] = 0
            var += (wls(Xv, y, w) - b) ** 2
        store[y_name] = {'b': b.tolist(), 'v': var.tolist()}
        json.dump(store, open(ck + '.tmp', 'w')); os.replace(ck + '.tmp', ck)
    est = np.array([store[y]['b'] for y in ys]); within = np.mean([store[y]['v'] for y in ys], 0)
    beta = est.mean(0)
    between = est.var(0, ddof=1) if len(ys) > 1 else 0
    se = np.sqrt(within + ((1 + 1 / len(ys)) * between if len(ys) > 1 else 0))
    def r2(Xm, b, y):
        ym = np.average(y, weights=w0)
        return 1 - np.average((y - Xm @ b) ** 2, weights=w0) / np.average((y - ym) ** 2, weights=w0)
    Xf = np.column_stack([np.ones(len(df)), fe.to_numpy()])
    r2_full = np.mean([r2(Xv, np.array(store[y]['b']), df[y].to_numpy('float64')) for y in ys])
    r2_fe = np.mean([r2(Xf, wls(Xf, df[y].to_numpy('float64'), w0), df[y].to_numpy('float64')) for y in ys])
    r = pd.DataFrame({'coef': beta, 'SE': se}, index=names).loc[preds]
    r['t'] = r.coef / r.SE; r['R2'] = r2_full; r['R2_country_FE_only'] = r2_fe
    r['n'] = len(df); r['model'] = label.split('_')[-1]
    return r

MODELS = {'M0': IND,                                                   # process indicators only
          'M1': IND + ['fd_score'],                                    # + Farm Drone score
          'M2': IND + ['fd_score', 'second_time'],                     # + effort on the second module
          'M3': IND + ['fd_score', 'second_time', 'fd_time_sq']}       # + squared Farm Drone time
OUTS = {'CT_second_module': ['ct_second'], 'CIL': CIL_SD}
results = []
for on, ys in OUTS.items():
    df = analytic.dropna(subset=MODELS['M3'] + ys[:1]).reset_index(drop=True)   # same sample for M0-M3
    for mn, preds in MODELS.items():
        r = fit(df, preds, ys, f'{on}_{mn}')
        r['outcome'] = on
        results.append(r)
        print(on, mn, 'n =', len(df), 'done')
res = pd.concat(results)
res.index.name = 'predictor'
res = res.reset_index().round(3)
res.to_csv(os.path.join(OUT_DIR, '03_stepwise_models.csv'), index=False)
res.pivot_table(index=['outcome', 'predictor'], columns='model', values='coef', sort=False)

# ===========================================================================
# EXPLORATORY
# ===========================================================================
# (a) M2 + reset x score interaction; (b) reset rate split by task group; (c) M2 within Farm Drone score tertiles
P2 = MODELS['M2']
EXTRA = []
def run(df, preds, ys, label, spec):
    r = fit(df.reset_index(drop=True), preds, ys, label)
    r['analysis'] = spec; r['outcome'] = label.split('__')[0]
    EXTRA.append(r); print(label, 'n =', len(df))

for on, ys in OUTS.items():
    d = analytic.dropna(subset=P2 + ['reset_x_score'] + ys[:1])
    run(d, P2 + ['reset_x_score'], ys, f'{on}__interaction', 'a. M2 + reset x Farm Drone score')
    pg = [p for p in P2 if p != 'reset_rate'] + RG
    d = analytic.dropna(subset=pg + ys[:1])
    run(d, pg, ys, f'{on}__task_group', 'b. reset rate by task group')
    d = analytic.dropna(subset=P2 + ys[:1]).copy()
    d['tertile'] = pd.qcut(d['fd_score'].rank(method='first'), 3, labels=['low', 'middle', 'high'])
    for tt in ['low', 'middle', 'high']:
        run(d[d.tertile == tt], P2, ys, f'{on}__tertile_{tt}', f'c. Farm Drone score tertile: {tt}')

ext = pd.concat(EXTRA)
ext.index.name = 'predictor'
ext = ext.reset_index().round(3)
ext.to_csv(os.path.join(OUT_DIR, '05_exploratory.csv'), index=False)
ext[ext.predictor.isin(['reset_rate', 'reset_x_score'] + RG)][['outcome', 'analysis', 'predictor', 'coef', 'SE', 't', 'n']]

# ===========================================================================
# BY_SYSTEM
# ===========================================================================
# Model M2 estimated separately in each education system (system-specific jackknife zones)
per = []
d0 = analytic.dropna(subset=P2 + ['ct_second'])
for c, g in d0.groupby(cid, observed=True):
    if len(g) < 300:
        continue
    g = g.reset_index(drop=True)
    Z = g[P2].astype('float64'); Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)
    Xv = np.column_stack([np.ones(len(g)), Z.to_numpy()])
    y = g['ct_second'].to_numpy('float64'); w0 = g['TOTWGTS'].to_numpy('float64')
    b = wls(Xv, y, w0); var = np.zeros_like(b)
    zones = g['JKZONES'].to_numpy(); reps = g['JKREPS'].to_numpy()
    for h in np.unique(zones[~np.isnan(zones)]):
        w = w0.copy(); inz = zones == h
        w[inz & (reps == 1)] *= 2; w[inz & (reps != 1)] = 0
        var += (wls(Xv, y, w) - b) ** 2
    row = {'system': str(c), 'n': len(g)}
    for j, p in enumerate(P2, 1):
        row[p] = round(b[j], 3)
        row[p + '_t'] = round(b[j] / np.sqrt(var[j]), 2) if var[j] > 0 else np.nan
    per.append(row)
per = pd.DataFrame(per)
per.to_csv(os.path.join(OUT_DIR, '04_by_system.csv'), index=False)

def tally(p):
    neg, pos, sig = per[p] < 0, per[p] > 0, per[p + '_t'].abs() > 2
    return (f'{p:12s}: negative {int(neg.sum())}/{len(per)} | negative & |t|>2: {int((neg & sig).sum())} '
            f'| positive & |t|>2: {int((pos & sig).sum())}')
TALLY = '\n'.join(tally(p) for p in IND)
print(TALLY)
per

# ===========================================================================
# SUPPLEMENTARY
# ===========================================================================
# (1) Sample by system and distribution of the second module
g = analytic.groupby(cid, observed=True)
samp = pd.DataFrame({'schools': g['IDSCHOOL'].nunique(), 'students_farm_drone': g.size(),
                     'analytic_n_M2': analytic.dropna(subset=P2 + ['ct_second']).groupby(cid, observed=True).size()})
samp = samp.join(pd.crosstab(analytic[cid], analytic['second_module'], normalize='index').mul(100).round(1))
samp.to_csv(os.path.join(OUT_DIR, '06_sample_by_system.csv'))

# (2) Balance across the randomly assigned second module (country FE, jackknife)
bal_vars = [v for v in ['female', 'SES', 'fd_score', 'fd_time', 'reset_rate', 'rapid_prop', 'n_tasks'] if v in analytic]
bal = []
d0 = analytic.dropna(subset=['second_module'])
for v in bal_vars:
    d = d0.dropna(subset=[v]).reset_index(drop=True)
    row = {'variable': v}
    for mod in OTHER_MODULES:
        dm = d[d.second_module == mod]
        row[mod] = round(np.average(dm[v], weights=dm['TOTWGTS']), 3)
    dd = pd.get_dummies(d['second_module'], drop_first=True, dtype='float64')
    fe = pd.get_dummies(d[cid].astype(str), drop_first=True, dtype='float64')
    Xv = np.column_stack([np.ones(len(d)), dd.to_numpy(), fe.to_numpy()])
    y = d[v].to_numpy('float64'); w0 = d['TOTWGTS'].to_numpy('float64')
    b = wls(Xv, y, w0); var = np.zeros_like(b)
    zones = d['JKZONES'].to_numpy(); reps = d['JKREPS'].to_numpy()
    for h in np.unique(zones[~np.isnan(zones)]):
        w = w0.copy(); inz = zones == h
        w[inz & (reps == 1)] *= 2; w[inz & (reps != 1)] = 0
        var += (wls(Xv, y, w) - b) ** 2
    k = dd.shape[1]
    sdv = np.sqrt(np.average((y - np.average(y, weights=w0)) ** 2, weights=w0))
    row['max_diff_SD'] = round(float(np.max(np.abs(b[1:1 + k]))) / sdv, 3)
    row['max_abs_t'] = round(float(np.max(np.abs(b[1:1 + k] / np.sqrt(var[1:1 + k])))), 2)
    bal.append(row)
bal = pd.DataFrame(bal)
bal.to_csv(os.path.join(OUT_DIR, '07_rotation_balance.csv'), index=False)

# (3) Robustness: M2 + gender + SES + second-module fixed effects
dmod = pd.get_dummies(analytic['second_module'], prefix='mod', drop_first=True, dtype='float64')
for c in dmod:
    analytic[c] = dmod[c]
ROB = P2 + [v for v in ['female', 'SES'] if v in analytic] + list(dmod.columns)
rob = []
for on, ys in OUTS.items():
    d = analytic.dropna(subset=ROB + ys[:1]).reset_index(drop=True)
    r = fit(d, ROB, ys, f'{on}_ROB'); r['outcome'] = on
    rob.append(r); print(on, 'robustness n =', len(d))
rob = pd.concat(rob); rob.index.name = 'predictor'
rob = rob.reset_index().round(3)
rob.to_csv(os.path.join(OUT_DIR, '08_robustness_gender_ses.csv'), index=False)
print(samp.to_string()); print(bal.to_string(index=False))
rob

# ===========================================================================
# SUMMARY
# ===========================================================================
tab = res.pivot_table(index=['outcome', 'predictor'], columns='model', values=['coef', 't'], sort=False)
tab.columns = [f'{m}_{s}' for s, m in tab.columns]
tab = tab[[f'{m}_{s}' for m in MODELS for s in ('coef', 't')]].round(3)
r2 = res.groupby(['outcome', 'model'])[['R2', 'R2_country_FE_only']].first().round(3)
focus = ['reset_rate', 'reset_x_score'] + RG
summary = f"""SUMMARY - Farm Drone process indicators and CT on the second module (ICILS 2023)
Systems: {analytic[cid].nunique()} | Farm Drone students: {len(analytic)} | with second CT module: {int(analytic.ct_second.notna().sum())}
SES variable: {SESVAR} | reset winsorising cap: {cap:.2f}/min | mean rapid-response proportion: {analytic.rapid_prop.mean():.3f}

STEPWISE MODELS (SD of outcome per SD of predictor; country FE; JK2; Rubin's rules)
M0 = process only | M1 = + Farm Drone score | M2 = + second-module time | M3 = + squared Farm Drone time
{tab.to_string()}

R2:
{r2.to_string()}

EXPLORATORY (reset rate):
{ext[ext.predictor.isin(focus)][['outcome', 'analysis', 'predictor', 'coef', 'SE', 't', 'n']].to_string(index=False)}

BY SYSTEM (CT on second module, M2):
{TALLY}

ROTATION BALANCE:
{bal.to_string(index=False)}

ROBUSTNESS (M2 + gender + SES + second-module FE):
{rob[rob.predictor.isin(IND + ['female', 'SES'])][['outcome', 'predictor', 'coef', 'SE', 't', 'n']].to_string(index=False)}
"""
print(summary)
open(os.path.join(OUT_DIR, '00_SUMMARY.txt'), 'w').write(summary)
print('Results in:', OUT_DIR)
