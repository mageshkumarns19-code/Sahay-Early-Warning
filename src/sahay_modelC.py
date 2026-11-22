# %% [markdown]
# # SAHAY – Model 5: Dynamic Distress Escalation Prediction
# ### Synthetic longitudinal cohort of victims of atrocities (SC/ST PoA Act context)
#
# **Research question (RQ1):** Do *legal-process events* (hearings, bail of the accused, threats,
# compensation delays, relocation) improve prediction of psychological-distress escalation
# beyond psychological signals alone?
#
# **What this notebook does**
# 1. Generates a synthetic weekly cohort of victims (latent distress + case events + noisy check-ins).
# 2. Builds two feature sets: **PSY** (check-in signals + demographics) and **PSY+LEGAL**.
# 3. Trains Logistic Regression, XGBoost and an Attention-LSTM, plus the rule-based Dynamic Distress Score (DDS).
# 4. Evaluates with victim-grouped 5-fold CV, a held-out test set (3 seeds), a temporal holdout,
#    a paired bootstrap test of the ablation, early-warning lead time, fairness by group, and SHAP explanations.
#
# **Important:** all data is synthetic. Cohort composition, legal timelines and distress trajectories are grounded
# in NCRB *Crime in India*, the SC/ST (PoA) Amendment Rules 2016 and a trauma-trajectory meta-analysis (each
# parameter is tagged in section 1). Parameters with no published Indian figure are marked [A] and varied in a
# sensitivity analysis (section 10c). Results show model behaviour and feasibility, **not** real-world performance.
# Minors are excluded from the simulated cohort.
#
# **Kaggle settings:** Accelerator = GPU T4 x2 (XGBoost and the LSTM use one GPU), Internet not required.
# For the full run use *Save Version → Save & Run All (Commit)*.

# %%
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
try:
    import shap
except ImportError:
    os.system('pip install -q shap')
    import shap

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 10})

IN_KAGGLE = os.path.exists('/kaggle/working')
OUT = '/kaggle/working/outputs' if IN_KAGGLE else './outputs'
os.makedirs(OUT, exist_ok=True)

QUICK = os.environ.get('SAHAY_QUICK', '0') == '1'   # set to '1' for a fast smoke test

CFG = dict(
    N_VICTIMS      = 1500 if QUICK else 6000,
    N_WEEKS        = 104,          # 24 months of weekly check-ins
    HORIZON        = 2,            # predict escalation within the next 2 weeks (14 days)
    HIGH           = 70,           # 'High' band threshold on the 0-100 distress scale
    SEQ_LEN        = 12,           # weeks of history for the sequence model
    SEEDS          = [42] if QUICK else [42, 7, 2026],
    N_FOLDS        = 3 if QUICK else 5,
    TEST_FRAC      = 0.15,
    VAL_FRAC       = 0.15,
    ALERT_BUDGET   = 0.10,         # officials can review the top 10% of weekly risk scores
    TEMPORAL_CUT   = 78,           # train on months 1-18, test on months 19-24
    N_BOOT         = 200 if QUICK else 1000,
    XGB = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
               colsample_bytree=0.8, early_stopping_rounds=50),
    DL_EPOCHS      = 4 if QUICK else 50,
    DL_BATCH       = 512,
    DL_LR          = 1e-3,
    DL_PATIENCE    = 8,
    DL_HIDDEN      = 64,
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE} | Quick mode: {QUICK} | Output folder: {OUT}')
print(json.dumps({k: v for k, v in CFG.items() if k != 'XGB'}, indent=1))

# %% [markdown]
# ## 1. Simulation parameters
# Each parameter is tagged with its source. **[S]** = taken from a cited source; **[S~]** = derived from a cited
# source with an approximation (see the parameter table in the paper); **[A]** = modelling assumption with no
# published Indian figure, varied in the sensitivity analysis (section 10c).
# Sources: NCRB *Crime in India* 2022–2024; SC/ST (PoA) Amendment Rules 2016 (Rule 7(2), Rule 12(4), Annexure-I);
# Galatzer-Levy, Huang & Bonanno (2018), *Clinical Psychology Review* 63:41–55; Factly analysis of NCRB court data.

# %%
P = dict(
    # cohort composition ---------------------------------------------------------------
    # [S~] NCRB crimes against SCs: simple hurt 32.0%, criminal intimidation 9.2% (2022, all India);
    # rape 13.9%, murder 2.4%, arson 0.2% of cases for trial (2023, Table 7B.5, metropolitan cities);
    # residual = other IPC heads, SC/ST Act-only, outraging modesty, riots, kidnapping. Arson raised to 0.5%
    # so the group exists in the cohort. Verify against NCRB Table 7A.1 (all-India) before submission.
    offence_mix = {'grievous_hurt': .35, 'witness_intimidation': .09, 'rape_gangrape': .14,
                   'murder_family': .025, 'arson': .005, 'caste_violence_family': .39},
    # [S~] State shares of crimes against SCs, 2023: UP 15,130 of 57,789 (26%); Rajasthan, MP, Bihar ~8,500,
    # ~7,800, ~7,000 → Hindi-speaking States ~70%. Split of the remaining ~30% is approximate.
    language_mix = {'Hindi': .70, 'Telugu': .07, 'Marathi': .05, 'Odia': .05, 'Kannada': .04,
                    'Tamil': .04, 'Gujarati': .02, 'Punjabi': .02, 'Bengali': .01},
    channel_mix = {'IVRS': .50, 'Chatbot': .30, 'App': .20},                 # [A] design choice
    age_mix = {'18-30': .40, '31-45': .35, '46-60': .18, '60+': .07},          # [A]
    # distress trajectories ------------------------------------------------------------------
    # [S] pooled prevalence 65.7% resilience, 20.8% recovery, 10.6% chronic, 8.9% delayed (54 studies),
    # normalised to sum to 1. General trauma populations; atrocity victims under ongoing threat may be more
    # chronic → varied in the sensitivity analysis.
    trajectory_mix = {'resilient': .620, 'recovering': .196, 'chronic': .100, 'delayed': .084},
    offence_severity = {'rape_gangrape': 8, 'murder_family': 8, 'grievous_hurt': 2, 'arson': 3,
                        'witness_intimidation': 3, 'caste_violence_family': 4},
    p_livelihood_loss = .35, livelihood_effect = 5.0, support_effect = 10.0,
    sens_sd = .35,                      # individual sensitivity to stressors (log-normal sd)
    ar = .75,                           # weekly persistence of deviations (half-life about 2.4 weeks)
    noise_sd = 3.0,
    # legal process ----------------------------------------------------------------------
    # [S] Rule 7(2): charge sheet within 60 days (~9 weeks); [S~] NCRB year-end pendency of investigation
    # (14,634 cases in 2023) shows many exceed it → filed between week 4 and week 26.
    chargesheet_week = (4, 26),
    p_chargesheet = .85,                 # [S] charge-sheeting rate 85.1% (45,439 of 53,372 PoA cases, NCRB 2023)
    first_hearing_gap = 4,               # [A]
    # [A] no published figure for special courts; mean gap ~10 weeks. [S] court pendency >90% (94% in 2019),
    # so trials run past the 24-month window (no verdict is simulated).
    hearing_gap_shape = 2.0, hearing_gap_scale = 4.0,
    p_bail = .55, bail_week = (2, 40),   # [A]
    threat_base = .006, witness_threat_mult = 2.5, threat_bail_mult = 3.0,
    threat_hearing_mult = 2.0, threat_reloc_mult = .3,
    p_relocate_after_threat = .15,
    # [S] Rule 12(4): relief within 7 days (week 1); Annexure-I pays instalments at FIR/medical report and at
    # charge sheet (final instalment at trial conclusion, outside the window). [A] share paid on time and delay.
    inst1_due = 1, p_on_time = .30, delay_mean_weeks = 12,
    # [A] effects of events on latent distress (points on the 0-100 scale), informed by the secondary-
    # victimisation literature; their size is varied in the sensitivity analysis -------------------
    eff_hearing_t2 = 3.0, eff_hearing_t1 = 5.0, eff_hearing = 6.0,   # anticipatory + day-of effect
    eff_bail = 12.0, eff_threat = 15.0, eff_relocate = 5.0,
    eff_overdue_per_wk = .4, eff_overdue_cap = 8.0,
    # check-in behaviour ------------------------------------------------------------------
    resp_intercept = 1.2, resp_engage = 1.0, resp_distress = .30, resp_threat = 1.5,
    p_conceal = .15, conceal_factor = .70,   # some victims under-report before hearings
)
TRAJ_CURVES = {
    'resilient':  lambda t: 30 + 15 * np.exp(-t / 8),
    'recovering': lambda t: 35 + 30 * np.exp(-t / 30),
    'chronic':    lambda t: 58 + 4 * np.sin(t / 15),
    'delayed':    lambda t: 30 + 30 / (1 + np.exp(-(t - 40) / 8)),
}

# %% [markdown]
# ## 2. Synthetic cohort generator

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_cohort(n, T, seed):
    rng = np.random.default_rng(seed)
    def pick(d, size=n):
        k = list(d); p = np.array(list(d.values()), float); return rng.choice(k, size=size, p=p / p.sum())

    offence, lang, traj = pick(P['offence_mix']), pick(P['language_mix']), pick(P['trajectory_mix'])
    channel, age = pick(P['channel_mix']), pick(P['age_mix'])
    gender = np.where(offence == 'rape_gangrape', 'F', np.where(rng.random(n) < .45, 'F', 'M'))
    livelihood = rng.random(n) < P['p_livelihood_loss']
    support = rng.beta(2, 2, n)
    sens = rng.lognormal(0, P['sens_sd'], n)
    engage = rng.normal(0, 1, n)
    conceal = rng.random(n) < P['p_conceal']
    sev = np.array([P['offence_severity'][o] for o in offence], float)
    wit = np.where(offence == 'witness_intimidation', P['witness_threat_mult'], 1.0)

    t = np.arange(T)
    base = np.zeros((n, T))
    for name, f in TRAJ_CURVES.items():
        base[traj == name] = f(t)[None, :]
    base += sev[:, None] + P['livelihood_effect'] * livelihood[:, None] - P['support_effect'] * (support[:, None] - .5)

    # court hearings
    H = np.zeros((n, T + 3), bool)
    cs = rng.integers(*P['chargesheet_week'], n)
    charged = rng.random(n) < P['p_chargesheet']          # cases never charge-sheeted have no trial
    cs = np.where(charged, cs, T + 10)
    for i in range(n):
        if not charged[i]:
            continue
        w = cs[i] + P['first_hearing_gap']
        while w < T:
            H[i, w] = True
            w += int(round(rng.gamma(P['hearing_gap_shape'], P['hearing_gap_scale']))) + 2
    # bail of the accused
    bail = rng.random(n) < P['p_bail']
    bail_wk = rng.integers(*P['bail_week'], n)
    B = np.zeros((n, T), bool); B[np.where(bail)[0], bail_wk[bail]] = True
    # compensation instalments (FIR stage, charge-sheet stage)
    O = np.zeros((n, T), bool)
    for due in [np.full(n, P['inst1_due']), cs]:
        late = rng.random(n) >= P['p_on_time']
        delay = np.where(late, rng.exponential(P['delay_mean_weeks'], n).astype(int) + 1, 0)
        O |= late[:, None] & (t[None, :] >= due[:, None]) & (t[None, :] < (due + delay)[:, None])

    D = np.zeros((n, T)); TH = np.zeros((n, T), bool); R = np.zeros((n, T), bool)
    relocated = np.zeros(n, bool); od_run = np.zeros(n); dev = rng.normal(0, 5, n)
    for w in range(T):
        since_bail = w - bail_wk
        recent_bail = bail & (since_bail >= 0) & (since_bail < 12)
        near_hear = H[:, w:w + 3].any(1)
        rate = (P['threat_base'] * wit * np.where(recent_bail, P['threat_bail_mult'], 1)
                * np.where(near_hear, P['threat_hearing_mult'], 1) * np.where(relocated, P['threat_reloc_mult'], 1))
        TH[:, w] = rng.random(n) < rate
        recent_threat = TH[:, max(0, w - 3):w + 1].any(1)
        newly = (~relocated) & recent_threat & (rng.random(n) < P['p_relocate_after_threat'])
        R[:, w] = newly; relocated |= newly
        od_run = np.where(O[:, w], od_run + 1, 0)
        imp = (P['eff_hearing_t2'] * H[:, w + 2] + P['eff_hearing_t1'] * H[:, w + 1] + P['eff_hearing'] * H[:, w]
               + P['eff_bail'] * B[:, w] + P['eff_threat'] * TH[:, w] + P['eff_relocate'] * newly)
        dev = P['ar'] * dev + sens * imp + rng.normal(0, P['noise_sd'], n)
        D[:, w] = np.clip(base[:, w] + np.minimum(P['eff_overdue_per_wk'] * od_run, P['eff_overdue_cap']) + dev, 0, 100)
    H = H[:, :T]

    # observed weekly check-ins
    th_recent = np.zeros((n, T), bool)
    for k in range(3):
        th_recent[:, k:] |= TH[:, :T - k]
    p_resp = sigmoid(P['resp_intercept'] + P['resp_engage'] * engage[:, None]
                     - P['resp_distress'] * (D - 50) / 10 - P['resp_threat'] * th_recent)
    resp = rng.random((n, T)) < p_resp
    near_h = np.zeros((n, T), bool)
    for k in range(3):
        near_h[:, :T - k] |= H[:, k:]
    screen = np.clip(np.round(D / 100 * 12 + rng.normal(0, 1.5, (n, T))), 0, 12)
    screen = np.where(conceal[:, None] & near_h, np.round(screen * P['conceal_factor']), screen)
    text_neg = np.clip(sigmoid((D - 50) / 12) + rng.normal(0, .15, (n, T)), 0, 1)
    voice = (D - 45) / 30 + rng.normal(0, .8, (n, T))
    voice[channel != 'IVRS'] = np.nan
    latency = np.exp(rng.normal(1 + .012 * (D - 50), .6))
    for a in (screen, text_neg, voice, latency):
        a[~resp] = np.nan

    static = pd.DataFrame(dict(victim_id=np.arange(n), offence=offence, gender=gender, language=lang,
                               channel=channel, age_group=age, trajectory=traj,
                               livelihood_loss=livelihood.astype(int)))
    arrays = dict(D=D, H=H, B=B, O=O, TH=TH, R=R, resp=resp, screen=screen, text_neg=text_neg,
                  voice=voice, latency=latency, bail_wk=np.where(bail, bail_wk, -1))
    return static, arrays

t0 = time.time()
static, A = simulate_cohort(CFG['N_VICTIMS'], CFG['N_WEEKS'], seed=2026)
print(f"Simulated {CFG['N_VICTIMS']} victims x {CFG['N_WEEKS']} weeks in {time.time() - t0:.1f}s")
print('Check-in response rate:', round(A['resp'].mean(), 3))
print('Share of victim-weeks in High band (latent):', round((A['D'] >= CFG['HIGH']).mean(), 3))
static.head()

# %% [markdown]
# ## 3. Features and labels
# **Label:** the victim is *not* in the High band this week, and enters it (latent distress ≥ 70) within the next 2 weeks.
# Latent distress is never used as a feature; the models see only check-in data, demographics and case events.

# %%
def ffill(a):
    n, T = a.shape
    idx = np.where(~np.isnan(a), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(n)[:, None], idx]

def roll_sum(x, k):
    c = np.cumsum(x, axis=1, dtype=float); out = c.copy(); out[:, k:] -= c[:, :-k]; return out

def build_features(static, A, cfg):
    n, T = A['D'].shape
    t = np.broadcast_to(np.arange(T)[None, :], (n, T))
    F = {}
    # psychological / engagement signals ----------------------------------------------
    resp = A['resp'].astype(float)
    for k in ['screen', 'text_neg', 'voice', 'latency']:
        F[f'{k}_last'] = ffill(A[k])
    s = A['screen']; obs = ~np.isnan(s); s0 = np.nan_to_num(s)
    cnt4 = roll_sum(obs, 4)
    F['screen_mean4'] = np.where(cnt4 > 0, roll_sum(s0, 4) / np.maximum(cnt4, 1), np.nan)
    sq = roll_sum(s0 ** 2, 4)
    F['screen_sd4'] = np.where(cnt4 > 1, np.sqrt(np.maximum(sq / np.maximum(cnt4, 1) - F['screen_mean4'] ** 2, 0)), np.nan)
    lag4 = np.full_like(F['screen_last'], np.nan); lag4[:, 4:] = F['screen_last'][:, :-4]
    F['screen_slope4'] = F['screen_last'] - lag4
    tn = A['text_neg']; tobs = ~np.isnan(tn)
    c4 = roll_sum(tobs, 4)
    F['text_mean4'] = np.where(c4 > 0, roll_sum(np.nan_to_num(tn), 4) / np.maximum(c4, 1), np.nan)
    F['responded'] = resp
    F['missed_last4'] = np.minimum(t + 1, 4) - roll_sum(resp, 4)
    last_resp = np.where(A['resp'], t, -1); np.maximum.accumulate(last_resp, axis=1, out=last_resp)
    F['weeks_since_response'] = np.where(last_resp >= 0, t - last_resp, t + 1)
    F['response_rate_cum'] = np.cumsum(resp, 1) / (t + 1)
    # legal-process events --------------------------------------------------------------
    H = A['H']
    Hp = np.concatenate([H, np.zeros((n, 3), bool)], 1)
    L = {}
    L['hearing_next1'] = Hp[:, 1:T + 1].astype(float)
    L['hearing_next2'] = (Hp[:, 1:T + 1] | Hp[:, 2:T + 2]).astype(float)
    nxt = np.full((n, T), 99.0); run = np.full(n, 99.0)
    for w in range(T - 1, -1, -1):
        nxt[:, w] = run
        run = np.where(H[:, w], 0, run + 1)
    L['weeks_to_next_hearing'] = np.minimum(nxt, 26)
    L['hearings_so_far'] = np.cumsum(H, 1).astype(float)
    bw = A['bail_wk'][:, None]
    granted = (bw >= 0) & (t >= bw)
    L['bail_granted'] = granted.astype(float)
    L['weeks_since_bail'] = np.where(granted, np.minimum(t - bw, 52), 60).astype(float)
    L['bail_recent12'] = (granted & (t - bw < 12)).astype(float)
    L['threats_last4'] = roll_sum(A['TH'], 4)
    L['threats_last12'] = roll_sum(A['TH'], 12)
    lt = np.where(A['TH'], t, -1); np.maximum.accumulate(lt, axis=1, out=lt)
    L['weeks_since_threat'] = np.where(lt >= 0, np.minimum(t - lt, 52), 60).astype(float)
    L['comp_overdue'] = A['O'].astype(float)
    od = np.zeros((n, T)); run = np.zeros(n)
    for w in range(T):
        run = np.where(A['O'][:, w], run + 1, 0); od[:, w] = run
    L['comp_overdue_weeks'] = od
    L['relocated'] = np.maximum.accumulate(A['R'], axis=1).astype(float)
    # static + time -------------------------------------------------------------------------
    S = {'week': t.astype(float)}
    st = static.set_index('victim_id')
    S['gender_F'] = np.repeat((st['gender'] == 'F').values[:, None], T, 1).astype(float)
    S['livelihood_loss'] = np.repeat(st['livelihood_loss'].values[:, None], T, 1).astype(float)
    S['age_ord'] = np.repeat(st['age_group'].map({'18-30': 0, '31-45': 1, '46-60': 2, '60+': 3}).values[:, None], T, 1).astype(float)
    for col, prefix in [('offence', 'off'), ('language', 'lang'), ('channel', 'ch')]:
        for v in sorted(st[col].unique()):
            S[f'{prefix}_{v}'] = np.repeat((st[col] == v).values[:, None], T, 1).astype(float)

    # labels ------------------------------------------------------------------------------
    D, Hz, hi = A['D'], cfg['HORIZON'], cfg['HIGH']
    fut = np.full((n, T), -1.0)
    for k in range(1, Hz + 1):
        fut[:, :T - k] = np.maximum(fut[:, :T - k], D[:, k:])
    y = (fut >= hi).astype(int)
    eligible = (D < hi) & (t >= 1) & (t <= T - 1 - Hz)

    allf = {**S, **F, **L}
    df = pd.DataFrame({k: v.ravel().astype(np.float32) for k, v in allf.items()})
    df.insert(0, 'victim_id', np.repeat(np.arange(n), T))
    df['y'] = y.ravel(); df['eligible'] = eligible.ravel(); df['latent_D'] = D.ravel().astype(np.float32)
    df = df.merge(static[['victim_id', 'offence', 'gender', 'language', 'channel']], on='victim_id', how='left')
    return df, list(S), list(F), list(L)

t0 = time.time()
panel, STATIC_FEATS, PSY_DYN, LEGAL_FEATS = build_features(static, A, CFG)
PSY_FEATS = STATIC_FEATS + PSY_DYN
FULL_FEATS = PSY_FEATS + LEGAL_FEATS
data = panel[panel['eligible']].reset_index(drop=True)
print(f'Features built in {time.time() - t0:.1f}s')
print(f'PSY features: {len(PSY_FEATS)} | PSY+LEGAL features: {len(FULL_FEATS)}')
print(f'Eligible victim-weeks: {len(data):,} | positive (escalation) rate: {data.y.mean():.3f}')

# %% [markdown]
# ## 4. Baseline: rule-based Dynamic Distress Score (DDS)
# DDS = 100 × (0.35·S + 0.20·T + 0.15·V + 0.15·E + 0.15·C), as proposed in the system design.

# %%
def dds_score(d):
    S = d['screen_last'].fillna(6) / 12
    Tx = d['text_neg_last'].fillna(.5)
    V = sigmoid(d['voice_last'].fillna(0).values)
    E = d['missed_last4'] / 4
    C = np.clip(.4 * d['hearing_next2'] + .4 * (d['threats_last4'] > 0) + .2 * d['comp_overdue'], 0, 1)
    return 100 * (.35 * S + .20 * Tx + .15 * V + .15 * E + .15 * C)

data['dds'] = dds_score(data)

# %% [markdown]
# ## 5. Splits and helper functions
# Victim-level split: 85% development (for 5-fold GroupKFold CV) and 15% held-out test, stratified by offence.

# %%
vic = static[['victim_id', 'offence']]
dev_ids, test_ids = train_test_split(vic.victim_id, test_size=CFG['TEST_FRAC'], stratify=vic.offence, random_state=42)
dev = data[data.victim_id.isin(dev_ids)].reset_index(drop=True)
test = data[data.victim_id.isin(test_ids)].reset_index(drop=True)
print(f'Dev rows: {len(dev):,} ({dev.victim_id.nunique()} victims) | Test rows: {len(test):,} ({test.victim_id.nunique()} victims)')

def metrics(y, p, budget=CFG['ALERT_BUDGET']):
    thr = np.quantile(p, 1 - budget)
    flag = p >= thr
    tp = (flag & (y == 1)).sum()
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                Recall_at_budget=tp / max(y.sum(), 1), Precision_at_budget=tp / max(flag.sum(), 1))

def inner_split(df, seed, frac=.10):
    ids = df.victim_id.unique()
    rng = np.random.default_rng(seed); va = rng.choice(ids, int(len(ids) * frac), replace=False)
    m = df.victim_id.isin(va)
    return df[~m], df[m]

def fit_xgb(tr, va, feats, seed):
    m = xgb.XGBClassifier(**CFG['XGB'], eval_metric='aucpr', tree_method='hist', device=DEVICE,
                          random_state=seed, n_jobs=-1)
    m.fit(tr[feats], tr.y, eval_set=[(va[feats], va.y)], verbose=False)
    return m

def fit_lr(tr, feats, seed):
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                      LogisticRegression(max_iter=2000, random_state=seed))
    return m.fit(tr[feats], tr.y)

# %% [markdown]
# ## 6. Victim-grouped 5-fold cross-validation (development set)

# %%
cv_rows = []
gkf = GroupKFold(n_splits=CFG['N_FOLDS'])
t0 = time.time()
for fold, (tri, vai) in enumerate(gkf.split(dev, dev.y, dev.victim_id)):
    tr_all, te = dev.iloc[tri], dev.iloc[vai]
    tr, es = inner_split(tr_all, seed=fold)
    cv_rows.append(dict(model='DDS (rule-based)', features='PSY+LEGAL', fold=fold, **metrics(te.y.values, te.dds.values)))
    for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
        lr = fit_lr(tr_all, feats, 42)
        cv_rows.append(dict(model='Logistic Regression', features=fname, fold=fold,
                            **metrics(te.y.values, lr.predict_proba(te[feats])[:, 1])))
        xm = fit_xgb(tr, es, feats, 42)
        cv_rows.append(dict(model='XGBoost', features=fname, fold=fold,
                            **metrics(te.y.values, xm.predict_proba(te[feats])[:, 1])))
    print(f'Fold {fold + 1}/{CFG["N_FOLDS"]} done ({time.time() - t0:.0f}s)')

cv = pd.DataFrame(cv_rows)
cv.to_csv(f'{OUT}/cv_fold_results.csv', index=False)
cv_summary = cv.groupby(['model', 'features'])[['AUROC', 'AUPRC', 'Recall_at_budget', 'Precision_at_budget']].agg(['mean', 'std']).round(4)
cv_summary.to_csv(f'{OUT}/cv_summary.csv')
cv_summary

# %% [markdown]
# ## 7. Attention-LSTM sequence model
# Input: the last 12 weeks of features for each victim-week (known-future hearing dates are included as covariates).
# Batches are gathered directly on the GPU for speed.

# %%
class AttnLSTM(nn.Module):
    def __init__(self, n_feat, h):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, h, batch_first=True)
        self.att = nn.Linear(h, 1)
        self.head = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        a = torch.softmax(self.att(o).squeeze(-1), dim=1)
        ctx = (a.unsqueeze(-1) * o).sum(1)
        return self.head(torch.cat([ctx, o[:, -1]], 1)).squeeze(-1), a

def make_tensor(panel, feats, mu, sd, n, T, L):
    X = ((panel[feats].values - mu) / sd).astype(np.float32)
    X = np.nan_to_num(X).reshape(n, T, len(feats))
    X = np.concatenate([np.zeros((n, L - 1, len(feats)), np.float32), X], 1)   # left-pad history
    return torch.tensor(X, device=DEVICE)

def gather(X, vids, weeks, L):
    v = torch.as_tensor(vids, device=DEVICE); w = torch.as_tensor(weeks, device=DEVICE)
    idx = w[:, None] + torch.arange(L, device=DEVICE)[None, :]
    return X[v[:, None], idx]

def predict_dl(model, X, df, L, bs=4096):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(df), bs):
            b = df.iloc[i:i + bs]
            out.append(torch.sigmoid(model(gather(X, b.victim_id.values, b.week.values.astype(int), L))[0]).cpu().numpy())
    return np.concatenate(out)

def train_dl(feats, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n, T, L = CFG['N_VICTIMS'], CFG['N_WEEKS'], CFG['SEQ_LEN']
    ref = panel[panel.victim_id.isin(tr.victim_id.unique())]
    mu, sd = ref[feats].mean().values, ref[feats].std().replace(0, 1).fillna(1).values
    X = make_tensor(panel, feats, mu, sd, n, T, L)
    model = AttnLSTM(len(feats), CFG['DL_HIDDEN']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG['DL_LR'])
    pw = torch.tensor(min((1 - tr.y.mean()) / tr.y.mean(), 10.0), device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yv = torch.tensor(tr.y.values, dtype=torch.float32, device=DEVICE)
    vids, weeks = tr.victim_id.values, tr.week.values.astype(int)
    best, best_state, wait = -1, None, 0
    for ep in range(CFG['DL_EPOCHS']):
        model.train(); perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), CFG['DL_BATCH']):
            b = perm[i:i + CFG['DL_BATCH']]
            logit, _ = model(gather(X, vids[b], weeks[b], L))
            loss = lossf(logit, yv[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ap = average_precision_score(va.y, predict_dl(model, X, va, L))
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f'  epoch {ep + 1:02d}  loss {loss.item():.4f}  val AUPRC {ap:.4f}')
        if wait >= CFG['DL_PATIENCE']:
            break
    model.load_state_dict(best_state)
    return model, X

# %% [markdown]
# ## 8. Final evaluation on the held-out test set (3 seeds)
# Development victims are split 70/15 (train / validation for early stopping); the 15% test victims are untouched.

# %%
dev_tr_ids, dev_va_ids = train_test_split(dev.victim_id.unique(), test_size=CFG['VAL_FRAC'] / (1 - CFG['TEST_FRAC']), random_state=42)
tr, va = dev[dev.victim_id.isin(dev_tr_ids)], dev[dev.victim_id.isin(dev_va_ids)]

test_rows, preds = [], {}
test_rows.append(dict(model='DDS (rule-based)', features='PSY+LEGAL', seed=0, **metrics(test.y.values, test.dds.values)))
preds[('DDS (rule-based)', 'PSY+LEGAL')] = test.dds.values / 100
final_xgb = {}
for seed in CFG['SEEDS']:
    for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
        lr = fit_lr(pd.concat([tr, va]), feats, seed)
        p = lr.predict_proba(test[feats])[:, 1]
        test_rows.append(dict(model='Logistic Regression', features=fname, seed=seed, **metrics(test.y.values, p)))
        xm = fit_xgb(tr, va, feats, seed)
        p = xm.predict_proba(test[feats])[:, 1]
        test_rows.append(dict(model='XGBoost', features=fname, seed=seed, **metrics(test.y.values, p)))
        if seed == CFG['SEEDS'][0]:
            preds[('Logistic Regression', fname)] = lr.predict_proba(test[feats])[:, 1]
            preds[('XGBoost', fname)] = p
            final_xgb[fname] = xm
        print(f'[seed {seed}] Attention-LSTM, {fname}')
        dl, Xt = train_dl(feats, tr, va, seed)
        p = predict_dl(dl, Xt, test, CFG['SEQ_LEN'])
        test_rows.append(dict(model='Attention-LSTM', features=fname, seed=seed, **metrics(test.y.values, p)))
        if seed == CFG['SEEDS'][0]:
            preds[('Attention-LSTM', fname)] = p
        del Xt
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

test_res = pd.DataFrame(test_rows)
test_res.to_csv(f'{OUT}/test_results_by_seed.csv', index=False)
test_summary = test_res.groupby(['model', 'features'])[['AUROC', 'AUPRC', 'Recall_at_budget', 'Precision_at_budget']].agg(['mean', 'std']).round(4)
test_summary.to_csv(f'{OUT}/test_summary.csv')
test_summary

# %% [markdown]
# ## 9. Ablation significance: paired bootstrap over victims (XGBoost, PSY vs PSY+LEGAL)

# %%
def boot_diff(y, p1, p2, groups, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    d_auc, d_ap = [], []
    for _ in range(n_boot):
        w = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug))[inv]
        if y[w > 0].min() == y[w > 0].max():
            continue
        d_auc.append(roc_auc_score(y, p2, sample_weight=w) - roc_auc_score(y, p1, sample_weight=w))
        d_ap.append(average_precision_score(y, p2, sample_weight=w) - average_precision_score(y, p1, sample_weight=w))
    out = {}
    for name, d in [('AUROC', d_auc), ('AUPRC', d_ap)]:
        d = np.array(d)
        out[name] = dict(delta=d.mean(), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                         p_value=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return pd.DataFrame(out).T

abl = []
for mname in ['Logistic Regression', 'XGBoost', 'Attention-LSTM']:
    r = boot_diff(test.y.values, preds[(mname, 'PSY')], preds[(mname, 'PSY+LEGAL')], test.victim_id.values, CFG['N_BOOT'])
    r.insert(0, 'model', mname); abl.append(r)
ablation = pd.concat(abl).round(4)
ablation.to_csv(f'{OUT}/ablation_bootstrap.csv')
ablation

# %% [markdown]
# ## 10. Temporal holdout (train months 1-18, test months 19-24 on unseen victims)

# %%
cut, Hz = CFG['TEMPORAL_CUT'], CFG['HORIZON']
tr_t = tr[tr.week + Hz < cut]; va_t = va[va.week + Hz < cut]; te_t = test[test.week >= cut]
temporal = []
for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
    xm = fit_xgb(tr_t, va_t, feats, 42)
    temporal.append(dict(model='XGBoost', features=fname, **metrics(te_t.y.values, xm.predict_proba(te_t[feats])[:, 1])))
temporal = pd.DataFrame(temporal).round(4)
temporal.to_csv(f'{OUT}/temporal_holdout.csv', index=False)
temporal

# %% [markdown]
# ## 10b. Negative control: legal events have no effect on distress
# The simulator builds in legal-event effects, so a reviewer may ask whether the ablation gain is circular.
# Here we re-simulate with every legal-event effect set to zero (events still happen and are still observed).
# If the pipeline is sound, adding legal features should give **no** meaningful gain in this cohort.

# %%
EFFECT_KEYS = ['eff_hearing_t2', 'eff_hearing_t1', 'eff_hearing', 'eff_bail', 'eff_threat',
               'eff_relocate', 'eff_overdue_per_wk', 'eff_overdue_cap', 'resp_threat', 'p_conceal']
saved = {k: P[k] for k in EFFECT_KEYS}
for k in EFFECT_KEYS:
    P[k] = 0.0
static0, A0 = simulate_cohort(CFG['N_VICTIMS'], CFG['N_WEEKS'], seed=2026)
for k, v in saved.items():
    P[k] = v
panel0, *_ = build_features(static0, A0, CFG)
data0 = panel0[panel0['eligible']].reset_index(drop=True)
d0_tr = data0[data0.victim_id.isin(dev_tr_ids)]; d0_va = data0[data0.victim_id.isin(dev_va_ids)]
d0_te = data0[data0.victim_id.isin(test_ids)].reset_index(drop=True)
nc = {}
for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
    nc[fname] = fit_xgb(d0_tr, d0_va, feats, 42).predict_proba(d0_te[feats])[:, 1]
neg_control = boot_diff(d0_te.y.values, nc['PSY'], nc['PSY+LEGAL'], d0_te.victim_id.values, CFG['N_BOOT']).round(4)
neg_control.insert(0, 'experiment', 'negative control (no legal effects)')
neg_control.to_csv(f'{OUT}/negative_control.csv')
print(f'Positive rate in negative-control cohort: {data0.y.mean():.3f}')
neg_control

# %% [markdown]
# ## 10c. Sensitivity analysis: parameters with no published Indian figure
# Each row re-simulates the cohort with one assumption changed (same seed and victim split) and retrains XGBoost
# (seed 42) with and without legal features. If the legal-feature gain stays positive across rows, the main
# conclusion does not hinge on these assumptions.

# %%
EFF_KEYS = ['eff_hearing_t2', 'eff_hearing_t1', 'eff_hearing', 'eff_bail', 'eff_threat', 'eff_relocate',
            'eff_overdue_per_wk', 'eff_overdue_cap']
SENS = [
    ('Base (as above)', {}),
    ('Hearing gap ~6 weeks', dict(hearing_gap_scale=2.0)),
    ('Hearing gap ~16 weeks', dict(hearing_gap_scale=7.0)),
    ('Relief mostly on time (70%, 4-wk delay)', dict(p_on_time=.70, delay_mean_weeks=4)),
    ('Relief mostly late (10%, 24-wk delay)', dict(p_on_time=.10, delay_mean_weeks=24)),
    ('More chronic distress (45/20/25/10)', dict(trajectory_mix={'resilient': .45, 'recovering': .20,
                                                               'chronic': .25, 'delayed': .10})),
    ('Bail rarely granted (30%)', dict(p_bail=.30)),
    ('Legal-event effects halved', {k: P[k] * .5 for k in EFF_KEYS}),
    ('Legal-event effects doubled', {k: P[k] * 2 for k in EFF_KEYS}),
]
sens_rows = []
for label, over in SENS:
    saved = {k: P[k] for k in over}
    P.update(over)
    st_s, A_s = simulate_cohort(CFG['N_VICTIMS'], CFG['N_WEEKS'], seed=2026)
    P.update(saved)
    pnl_s, *_ = build_features(st_s, A_s, CFG)
    d_s = pnl_s[pnl_s['eligible']].reset_index(drop=True)
    s_tr, s_va = d_s[d_s.victim_id.isin(dev_tr_ids)], d_s[d_s.victim_id.isin(dev_va_ids)]
    s_te = d_s[d_s.victim_id.isin(test_ids)].reset_index(drop=True)
    r = dict(scenario=label, positive_rate=round(float(d_s.y.mean()), 4))
    for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
        r[f'AUPRC_{fname}'] = average_precision_score(s_te.y, fit_xgb(s_tr, s_va, feats, 42).predict_proba(s_te[feats])[:, 1])
    r['legal_gain_AUPRC'] = r['AUPRC_PSY+LEGAL'] - r['AUPRC_PSY']
    sens_rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
sensitivity = pd.DataFrame(sens_rows).round(4)
sensitivity.to_csv(f'{OUT}/sensitivity_analysis.csv', index=False)

fig, ax = plt.subplots(figsize=(8, .45 * len(sensitivity) + 1))
yy = np.arange(len(sensitivity))[::-1]
ax.hlines(yy, sensitivity['AUPRC_PSY'], sensitivity['AUPRC_PSY+LEGAL'], color='#c9ced6', lw=3)
ax.plot(sensitivity['AUPRC_PSY'], yy, 'o', color='#8a8f98', label='PSY')
ax.plot(sensitivity['AUPRC_PSY+LEGAL'], yy, 'o', color='#2a6fdb', label='PSY+LEGAL')
ax.set_yticks(yy, sensitivity.scenario); ax.set_xlabel('Test AUPRC (XGBoost, seed 42)')
ax.set_title('Sensitivity of the legal-feature gain to unsourced assumptions')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_sensitivity.png', dpi=200); plt.show()
sensitivity

# %% [markdown]
# ## 11. Early warning: episode detection and lead time
# An *escalation episode* starts in the week latent distress first crosses 70. An episode counts as detected if the
# model flags the victim (top-10% alert budget) in any of the 4 weeks before onset.

# %%
def lead_time(df, p, budget=CFG['ALERT_BUDGET'], look=4):
    thr = np.quantile(p, 1 - budget)
    d = df[['victim_id', 'week']].copy(); d['flag'] = p >= thr
    flags = {(v, w): f for v, w, f in d.itertuples(index=False)}
    D = A['D']; det, leads, n_ep = 0, [], 0
    for v in df.victim_id.unique():
        x = D[v]
        onsets = np.where((x[1:] >= CFG['HIGH']) & (x[:-1] < CFG['HIGH']))[0] + 1
        for s in onsets[onsets >= look + 1]:
            n_ep += 1
            hit = [s - w for w in range(s - look, s) if flags.get((v, w), False)]
            if hit:
                det += 1; leads.append(max(hit))
    return dict(episodes=n_ep, detected_pct=100 * det / max(n_ep, 1),
                mean_lead_weeks=np.mean(leads) if leads else np.nan, median_lead_weeks=np.median(leads) if leads else np.nan)

lead = pd.DataFrame([dict(model=m, features=f, **lead_time(test, p)) for (m, f), p in preds.items()]).round(2)
lead.to_csv(f'{OUT}/lead_time.csv', index=False)
lead

# %% [markdown]
# ## 12. Fairness: recall and AUROC by gender, language and offence (XGBoost, PSY+LEGAL)

# %%
p_full = preds[('XGBoost', 'PSY+LEGAL')]
thr = np.quantile(p_full, 1 - CFG['ALERT_BUDGET'])
fair = []
for g in ['gender', 'language', 'offence', 'channel']:
    for val, idx in test.groupby(g).groups.items():
        yy, pp = test.y.values[idx], p_full[idx]
        if yy.sum() < 5 or yy.sum() == len(yy):
            continue
        fair.append(dict(attribute=g, group=val, n_rows=len(idx), positives=int(yy.sum()),
                         AUROC=roc_auc_score(yy, pp), Recall_at_threshold=((pp >= thr) & (yy == 1)).sum() / yy.sum()))
fair = pd.DataFrame(fair).round(4)
fair.to_csv(f'{OUT}/fairness.csv', index=False)
fair

# %% [markdown]
# ## 13. Explainability (SHAP) and a plain-language alert

# %%
LABELS = {
    'screen_last': 'latest screening score', 'screen_mean4': 'average screening score (4 wks)',
    'screen_slope4': 'rise in screening score (4 wks)', 'screen_sd4': 'variability of screening score',
    'text_neg_last': 'negative emotion in latest reply', 'text_mean4': 'negative emotion (4 wks)',
    'voice_last': 'voice stress in latest call', 'latency_last': 'slow reply to check-in',
    'missed_last4': 'missed check-ins (4 wks)', 'weeks_since_response': 'weeks since last reply',
    'response_rate_cum': 'overall response rate', 'responded': 'replied this week',
    'hearing_next1': 'court hearing next week', 'hearing_next2': 'court hearing within 2 weeks',
    'weeks_to_next_hearing': 'weeks until next hearing', 'hearings_so_far': 'number of hearings so far',
    'bail_granted': 'accused granted bail', 'weeks_since_bail': 'weeks since bail of accused',
    'bail_recent12': 'bail granted in last 12 weeks', 'threats_last4': 'threats reported (4 wks)',
    'threats_last12': 'threats reported (12 wks)', 'weeks_since_threat': 'weeks since last threat',
    'comp_overdue': 'relief instalment overdue', 'comp_overdue_weeks': 'weeks relief has been overdue',
    'relocated': 'relocated after threat', 'livelihood_loss': 'loss of livelihood', 'week': 'weeks since complaint',
}
model = final_xgb['PSY+LEGAL']
sample = test.sample(min(3000, len(test)), random_state=0)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(sample[FULL_FEATS])
imp = pd.Series(np.abs(sv).mean(0), index=FULL_FEATS).sort_values(ascending=False)
imp.rename('mean_abs_shap').to_csv(f'{OUT}/shap_importance.csv')
legal_share = imp[LEGAL_FEATS].sum() / imp.sum()
print(f'Share of total SHAP importance from legal-process features: {legal_share:.1%}')

plt.figure()
shap.summary_plot(sv, sample[FULL_FEATS].rename(columns=LABELS), max_display=15, show=False)
plt.title('SHAP summary – XGBoost (PSY+LEGAL)')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_shap_summary.png', dpi=200, bbox_inches='tight'); plt.show()

def explain_alert(row_df):
    pr = model.predict_proba(row_df[FULL_FEATS])[0, 1]
    s = pd.Series(explainer.shap_values(row_df[FULL_FEATS])[0], index=FULL_FEATS)
    top = s[s > 0].sort_values(ascending=False).head(4)
    lines = [f'Escalation risk in next 14 days: {pr:.0%}', 'Main reasons:']
    lines += [f'  - {LABELS.get(k, k)} (value {row_df[k].values[0]:.1f}, +{v:.2f} log-odds)' for k, v in top.items()]
    return '\n'.join(lines)

hi_row = test.iloc[[int(np.argmax(p_full))]]
alert_text = explain_alert(hi_row)
print(alert_text)
open(f'{OUT}/example_alert.txt', 'w').write(alert_text)

# %% [markdown]
# ## 14. Figures for the paper

# %%
COL = {'PSY': '#8a8f98', 'PSY+LEGAL': '#2a6fdb'}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for f in ['PSY', 'PSY+LEGAL']:
    p = preds[('XGBoost', f)]
    fpr, tpr, _ = roc_curve(test.y, p); pr, rc, _ = precision_recall_curve(test.y, p)
    ax[0].plot(fpr, tpr, color=COL[f], lw=2, label=f'{f} (AUROC {roc_auc_score(test.y, p):.3f})')
    ax[1].plot(rc, pr, color=COL[f], lw=2, label=f'{f} (AUPRC {average_precision_score(test.y, p):.3f})')
ax[0].plot([0, 1], [0, 1], ls='--', c='k', lw=.8)
ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC – XGBoost, test set')
ax[1].axhline(test.y.mean(), ls='--', c='k', lw=.8)
ax[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall – XGBoost, test set')
for a in ax: a.legend(loc='lower right' if a is ax[0] else 'upper right')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_roc_pr.png', dpi=200); plt.show()

m = test_res.groupby(['model', 'features']).AUPRC.agg(['mean', 'std']).reset_index()
models = ['Logistic Regression', 'XGBoost', 'Attention-LSTM']
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(models)); w = .36
for j, f in enumerate(['PSY', 'PSY+LEGAL']):
    r = m[m.features == f].set_index('model').reindex(models)
    ax.bar(x + (j - .5) * w, r['mean'], w, yerr=r['std'].fillna(0), color=COL[f], label=f, capsize=3)
dds_ap = test_res[test_res.model.str.startswith('DDS')].AUPRC.mean()
ax.axhline(dds_ap, ls=':', c='#c0392b', label=f'Rule-based DDS ({dds_ap:.3f})')
ax.set_xticks(x, models); ax.set_ylabel('AUPRC (mean ± SD over seeds)'); ax.set_title('Ablation: effect of legal-process features')
ax.set_ylim(0, m['mean'].max() * 1.35)
ax.legend(loc='upper left', ncol=3, fontsize=8, frameon=False); plt.tight_layout(); plt.savefig(f'{OUT}/fig_ablation.png', dpi=200); plt.show()

fs = fair[fair.attribute.isin(['gender', 'offence', 'language'])]
fig, ax = plt.subplots(figsize=(8, .32 * len(fs) + 1))
ax.barh(fs.attribute + ': ' + fs.group.astype(str), fs.Recall_at_threshold, color='#2a6fdb')
ax.axvline(((p_full >= thr) & (test.y == 1)).sum() / test.y.sum(), ls='--', c='k', lw=.8, label='overall')
ax.set_xlabel('Recall at alert threshold'); ax.set_title('Fairness check – XGBoost (PSY+LEGAL)'); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/fig_fairness.png', dpi=200); plt.show()

# example victim trajectory
cand = test.groupby('victim_id').agg(pos=('y', 'sum')).query('pos > 0').index
th_counts = A['TH'][cand].sum(1)
v = int(cand[np.argmax(th_counts)])
fig, ax = plt.subplots(figsize=(11, 4.2))
wk = np.arange(CFG['N_WEEKS'])
ax.plot(wk, A['D'][v], c='#333', lw=1.6, label='Latent distress (hidden)')
ax.scatter(wk, A['screen'][v] / 12 * 100, s=12, c='#8a8f98', label='Check-in screening (scaled)')
ax.axhline(CFG['HIGH'], ls='--', c='#c0392b', lw=.8)
for h in np.where(A['H'][v])[0]: ax.axvline(h, c='#2a6fdb', alpha=.35, lw=1)
for h in np.where(A['TH'][v])[0]: ax.scatter(h, 97, marker='v', c='#c0392b', s=40)
if A['bail_wk'][v] >= 0: ax.axvline(A['bail_wk'][v], c='#e67e22', lw=2, alpha=.7)
ax.fill_between(wk, 0, 100, where=A['O'][v], color='#f1c40f', alpha=.12, step='mid')
ax.set(xlabel='Week since complaint', ylabel='Distress (0-100)', ylim=(0, 100),
       title=f'Victim {v}: blue = hearings, orange = bail, red ▼ = threats, yellow = relief overdue')
rows = test[test.victim_id == v]
ax2 = ax.twinx(); ax2.grid(False)
for f in ['PSY', 'PSY+LEGAL']:
    pv = pd.Series(preds[('XGBoost', f)][rows.index], index=rows.week.astype(int)).reindex(wk)  # gaps = already in crisis
    ax2.plot(wk, pv.values, color=COL[f], lw=1.3, alpha=.9, label=f'Predicted risk ({f})')
ax2.set_ylabel('Predicted 14-day escalation risk'); ax2.set_ylim(0, 1)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_victim_trajectory.png', dpi=200); plt.show()

# %% [markdown]
# ## 15. Summary table for the paper

# %%
def fmt(df):
    g = df.groupby(['model', 'features'])
    rows = []
    for (mname, f), d in g:
        rows.append({'Model': mname, 'Features': f,
                     **{k: f"{d[k].mean():.3f} ± {d[k].std(ddof=1) if len(d) > 1 else 0:.3f}"
                        for k in ['AUROC', 'AUPRC', 'Recall_at_budget', 'Precision_at_budget']}})
    return pd.DataFrame(rows)

paper_table = fmt(test_res)
paper_table.to_csv(f'{OUT}/paper_table_test.csv', index=False)
summary = dict(config={k: v for k, v in CFG.items()}, n_eligible_rows=int(len(data)),
               positive_rate=float(data.y.mean()), legal_shap_share=float(legal_share),
               test=paper_table.to_dict('records'), ablation=ablation.reset_index().to_dict('records'),
               temporal=temporal.to_dict('records'), lead_time=lead.to_dict('records'),
               negative_control=neg_control.reset_index().to_dict('records'),
               sensitivity=sensitivity.to_dict('records'))
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
panel.drop(columns=['eligible']).to_csv(f'{OUT}/synthetic_cohort_panel.csv.gz', index=False, compression='gzip')
static.to_csv(f'{OUT}/synthetic_cohort_static.csv', index=False)
print('Saved outputs:', sorted(os.listdir(OUT)))
paper_table
