# %% [markdown]
# # SAHAY – Integration: Model A (text) + Model B (voice) → Model C (dynamic risk predictor)
# ### End-to-end test with realistic, imperfect text analysis
#
# In the Model C notebook, the check-in text signal came straight from the simulator, which is like assuming a
# perfect text model. Here the text signal is produced the way the deployed system would produce it:
# each check-in reply has a true emotional tone, and **Model A's measured confusion matrix** (DravidianCodeMix test
# set, multi-task MuRIL) decides what the system actually "hears". The risk predictor is then retrained on these
# imperfect signals.
#
# **Questions answered**
# 1. How much does risk prediction lose when the perfect text signal is replaced by Model A's real errors?
# 2. Does the legal-process gain (RQ1) survive realistic text errors?
# 3. How good does the text model need to be? (sensitivity sweep from perfect to random text)
# 4. How much do text and voice each contribute? (scenarios without text and without voice)
#
# **Model B slot:** when the voice model is trained, enter its sensitivity and specificity in `MODEL_B` and re-run;
# the "Model A + Model B" scenario is then added automatically.
#
# **Data:** synthetic cohort (same simulator and parameters as the Model C notebook; parameters grounded in NCRB, PoA Rules and trajectory literature; see section B). **Kaggle:** GPU T4 x2 (XGBoost uses one GPU), Internet off, no input
# datasets. Expected time: about 30–60 minutes.

# %%
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import torch

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 10})

IN_KAGGLE = os.path.exists('/kaggle/working')
OUT = '/kaggle/working/outputs_integration' if IN_KAGGLE else './outputs_integration'
os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get('SAHAY_QUICK', '0') == '1'

CFG = dict(
    N_VICTIMS    = 1500 if QUICK else 6000,
    N_WEEKS      = 104,
    HORIZON      = 2,
    HIGH         = 70,
    SEEDS        = [42] if QUICK else [42, 7, 2026],
    TEST_FRAC    = 0.15,
    ALERT_BUDGET = 0.10,
    N_BOOT       = 200 if QUICK else 1000,
    XGB = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
               colsample_bytree=0.8, early_stopping_rounds=50),
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE} | quick: {QUICK} | output: {OUT}')

# %% [markdown]
# ## A. Model A's measured errors and the Model B slot
# Rows = true class, columns = predicted class, row-normalised, from Model A (multi-task MuRIL, seed 42,
# DravidianCodeMix test set; values rounded to 2 decimals from `fig_confusion_matrices.png`).
# Replace with exact values from your own `outputs_modelA` if you re-run Model A.

# %%
SENT = ['Positive', 'Negative', 'Mixed_feelings', 'unknown_state', 'not_native']
MODEL_A_CM = np.array([
    [0.77, 0.05, 0.04, 0.11, 0.03],   # true Positive
    [0.20, 0.48, 0.14, 0.16, 0.02],   # true Negative
    [0.39, 0.16, 0.27, 0.15, 0.02],   # true Mixed_feelings
    [0.18, 0.09, 0.07, 0.63, 0.03],   # true unknown_state
    [0.24, 0.03, 0.01, 0.12, 0.60],   # true not_native
])
MODEL_A_CM = MODEL_A_CM / MODEL_A_CM.sum(1, keepdims=True)

# Score the risk model receives for each predicted class (higher = more negative tone).
CLASS_SCORE = {'Positive': 0.15, 'Negative': 0.85, 'Mixed_feelings': 0.55, 'unknown_state': 0.45, 'not_native': np.nan}

# How a reply's true tone depends on latent distress D (assumptions; tune or ground in pilot data).
TONE = dict(center=55.0, scale=8.0,            # P(negative vs positive) = sigmoid((D - center) / scale)
            mixed_peak=0.25, mixed_width=12.0,  # mixed feelings are most common near the centre
            p_unknown=0.12,                     # replies with no clear tone
            p_not_native=0.03)                  # replies in a language/script the model does not handle

# Model B (voice) slot: fill in after training the voice model, e.g. dict(sensitivity=0.70, specificity=0.75)
# for detecting distressed speech (latent D >= VOICE_THRESHOLD). Leave as None to skip that scenario.
MODEL_B = dict(sensitivity=0.823, specificity=0.889)   # from Model B (Androids, multi-task, speaker level)
VOICE_THRESHOLD = 60

# %% [markdown]
# ## B. Simulation parameters (identical to the Model C notebook)
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
# ## C. Synthetic cohort generator

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
# ## C2. Features and labels
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
# ## C3. Rule-based Dynamic Distress Score (DDS)
# DDS = 100 × (0.35·S + 0.20·T + 0.15·V + 0.15·E + 0.15·C), as proposed in the system design.

# %%
def dds_score(d):
    S = d['screen_last'].fillna(6) / 12
    Tx = d['text_neg_last'].fillna(.5)
    V = sigmoid(d['voice_last'].fillna(0).values)
    E = d['missed_last4'] / 4
    C = np.clip(.4 * d['hearing_next2'] + .4 * (d['threats_last4'] > 0) + .2 * d['comp_overdue'], 0, 1)
    return 100 * (.35 * S + .20 * Tx + .15 * V + .15 * E + .15 * C)


# %% [markdown]
# ## D. Splits and helper functions (same victim-level split as the Model C notebook)

# %%
vic = static[['victim_id', 'offence']]
dev_ids, test_ids = train_test_split(vic.victim_id, test_size=CFG['TEST_FRAC'], stratify=vic.offence, random_state=42)
dev_tr_ids, dev_va_ids = train_test_split(np.array(dev_ids), test_size=0.15 / (1 - CFG['TEST_FRAC']), random_state=42)

def metrics(y, p, budget=CFG['ALERT_BUDGET']):
    thr = np.quantile(p, 1 - budget); flag = p >= thr; tp = (flag & (y == 1)).sum()
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                Recall_at_budget=tp / max(y.sum(), 1), Precision_at_budget=tp / max(flag.sum(), 1))

def fit_xgb(tr, va, feats, seed):
    m = xgb.XGBClassifier(**CFG['XGB'], eval_metric='aucpr', tree_method='hist', device=DEVICE,
                          random_state=seed, n_jobs=-1)
    m.fit(tr[feats], tr.y, eval_set=[(va[feats], va.y)], verbose=False)
    return m

def boot_diff(y, p1, p2, groups, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    d_auc, d_ap = [], []
    for _ in range(n_boot):
        w = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug))[inv]
        d_auc.append(roc_auc_score(y, p2, sample_weight=w) - roc_auc_score(y, p1, sample_weight=w))
        d_ap.append(average_precision_score(y, p2, sample_weight=w) - average_precision_score(y, p1, sample_weight=w))
    out = {}
    for name, d in [('AUROC', d_auc), ('AUPRC', d_ap)]:
        d = np.array(d)
        out[f'delta_{name}'] = d.mean(); out[f'{name}_ci_low'] = np.percentile(d, 2.5)
        out[f'{name}_ci_high'] = np.percentile(d, 97.5)
        out[f'{name}_p'] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return out

def lead_time(df, p, budget=CFG['ALERT_BUDGET'], look=4):
    thr = np.quantile(p, 1 - budget)
    flags = {(v, w): f for v, w, f in zip(df.victim_id.values, df.week.values.astype(int), p >= thr)}
    D = A['D']; det, leads, n_ep = 0, [], 0
    for v in df.victim_id.unique():
        x = D[v]
        onsets = np.where((x[1:] >= CFG['HIGH']) & (x[:-1] < CFG['HIGH']))[0] + 1
        for s in onsets[onsets >= look + 1]:
            n_ep += 1
            hit = [s - w for w in range(s - look, s) if flags.get((v, w), False)]
            if hit: det += 1; leads.append(max(hit))
    return dict(episodes=n_ep, detected_pct=100 * det / max(n_ep, 1),
                median_lead_weeks=float(np.median(leads)) if leads else np.nan)

# %% [markdown]
# ## E. True tone of each check-in reply, and the text / voice channels

# %%
rng_tone = np.random.default_rng(11)
D, resp = A['D'], A['resp']
n, T = D.shape
u = rng_tone.random((4, n, T))
p_mixed = TONE['mixed_peak'] * np.exp(-((D - TONE['center']) / TONE['mixed_width']) ** 2)
p_neg = sigmoid((D - TONE['center']) / TONE['scale'])
true_tone = np.where(u[3] < p_neg, 1, 0)                     # Negative (1) or Positive (0)
true_tone = np.where(u[2] < p_mixed, 2, true_tone)           # Mixed_feelings
true_tone = np.where(u[1] < TONE['p_unknown'], 3, true_tone) # unknown_state
true_tone = np.where(u[0] < TONE['p_not_native'], 4, true_tone)  # not_native
prior = np.bincount(true_tone[resp], minlength=5) / resp.sum()
print('True tone distribution of replies:', dict(zip(SENT, prior.round(3))))
SCORES = np.array([CLASS_SCORE[c] for c in SENT], float)

def text_from_cm(cm, seed):
    """What the deployed system 'hears': sample Model A's prediction for each reply from its confusion matrix."""
    r = np.random.default_rng(seed)
    cum = np.cumsum(cm, 1)[true_tone]                       # (n, T, 5)
    pred = (r.random((n, T))[..., None] > cum).sum(-1).clip(0, 4)
    txt = SCORES[pred] + r.normal(0, 0.03, (n, T))
    txt[~resp] = np.nan
    return txt, pred

def empirical_macro_f1(pred):
    from sklearn.metrics import f1_score
    m = resp
    return f1_score(true_tone[m], pred[m], average='macro', labels=[0, 1, 2, 3, 4], zero_division=0)

def expected_macro_f1(cm):
    N = prior[:, None] * cm
    prec = np.diag(N) / np.maximum(N.sum(0), 1e-12); rec = np.diag(cm)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return float(f1[prior > 0].mean())

def voice_channel(mode, seed):
    if mode == 'simulated':
        return A['voice']
    v = np.full((n, T), np.nan)
    if mode == 'model_b':
        r = np.random.default_rng(seed)
        distressed = D >= VOICE_THRESHOLD; x = r.random((n, T))
        pred = np.where(distressed, x < MODEL_B['sensitivity'], x < 1 - MODEL_B['specificity'])
        v = np.where(pred, 1.0, -1.0) + r.normal(0, 0.3, (n, T))
        v[(static.channel != 'IVRS').values] = np.nan; v[~resp] = np.nan
    return v

print(f'Expected macro-F1 of Model A on the simulated replies: {expected_macro_f1(MODEL_A_CM):.3f}')

# %% [markdown]
# ## F. Scenarios: oracle vs perfect classifier vs Model A vs no text / no voice

# %%
I5 = np.eye(5)
SCENARIOS = [
    dict(name='Oracle text (simulator)', text='oracle', voice='simulated'),
    dict(name='Perfect tone classifier', text=I5, voice='simulated'),
    dict(name='Model A text (realistic)', text=MODEL_A_CM, voice='simulated'),
    dict(name='Model A text, no voice', text=MODEL_A_CM, voice='none'),
    dict(name='No text, simulated voice', text='none', voice='simulated'),
    # Free-text only: the victim replies in their own words but skips the screening questions
    dict(name='No screening: oracle text', text='oracle', voice='simulated', screen='none'),
    dict(name='No screening: Model A text', text=MODEL_A_CM, voice='simulated', screen='none'),
    dict(name='No screening, no text', text='none', voice='simulated', screen='none'),
]
N_WITH_SCREENING = 5
if MODEL_B['sensitivity'] is not None:
    SCENARIOS.insert(5, dict(name='Model A text + Model B voice', text=MODEL_A_CM, voice='model_b'))
    N_WITH_SCREENING = 6

def scenario_data(sc, seed=101):
    As = dict(A)
    if isinstance(sc['text'], str):
        As['text_neg'] = A['text_neg'] if sc['text'] == 'oracle' else np.full((n, T), np.nan)
        f1 = np.nan
    else:
        As['text_neg'], pred = text_from_cm(sc['text'], seed); f1 = empirical_macro_f1(pred)
    As['voice'] = voice_channel(sc['voice'], seed)
    if sc.get('screen', 'yes') == 'none':          # victim skips the structured screening questions
        As['screen'] = np.full((n, T), np.nan)
    pnl, *_ = build_features(static, As, CFG)
    d = pnl[pnl['eligible']].reset_index(drop=True)
    d['dds'] = dds_score(d)
    return d, f1

rows, preds, leads, abl = [], {}, [], []
for sc in SCENARIOS:
    t0 = time.time()
    d, f1 = scenario_data(sc)
    tr, va = d[d.victim_id.isin(dev_tr_ids)], d[d.victim_id.isin(dev_va_ids)]
    te = d[d.victim_id.isin(test_ids)].reset_index(drop=True)
    rows.append(dict(scenario=sc['name'], features='DDS (rule-based)', seed=0, text_macro_F1=f1,
                     **metrics(te.y.values, te.dds.values)))
    for seed in CFG['SEEDS']:
        for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
            p = fit_xgb(tr, va, feats, seed).predict_proba(te[feats])[:, 1]
            rows.append(dict(scenario=sc['name'], features=f'XGBoost {fname}', seed=seed, text_macro_F1=f1,
                             **metrics(te.y.values, p)))
            if seed == CFG['SEEDS'][0]:
                preds[(sc['name'], fname)] = p
    p1, p2 = preds[(sc['name'], 'PSY')], preds[(sc['name'], 'PSY+LEGAL')]
    abl.append(dict(scenario=sc['name'], **boot_diff(te.y.values, p1, p2, te.victim_id.values, CFG['N_BOOT'])))
    for fname in ['PSY', 'PSY+LEGAL']:
        leads.append(dict(scenario=sc['name'], features=fname, **lead_time(te, preds[(sc['name'], fname)])))
    print(f"{sc['name']:<32} text macro-F1 {f1 if f1 == f1 else float('nan'):.3f} | "
          f"AUPRC PSY {np.mean([r['AUPRC'] for r in rows if r['scenario'] == sc['name'] and r['features'] == 'XGBoost PSY']):.3f} "
          f"→ PSY+LEGAL {np.mean([r['AUPRC'] for r in rows if r['scenario'] == sc['name'] and r['features'] == 'XGBoost PSY+LEGAL']):.3f} "
          f"| {time.time() - t0:.0f}s")

res = pd.DataFrame(rows); res.to_csv(f'{OUT}/scenario_results_by_seed.csv', index=False)
summ = res.groupby(['scenario', 'features'], sort=False)[['text_macro_F1', 'AUROC', 'AUPRC', 'Recall_at_budget']].agg(['mean', 'std']).round(4)
summ.to_csv(f'{OUT}/scenario_summary.csv')
ablation = pd.DataFrame(abl).round(4); ablation.to_csv(f'{OUT}/ablation_by_scenario.csv', index=False)
lead = pd.DataFrame(leads).round(2); lead.to_csv(f'{OUT}/lead_time_by_scenario.csv', index=False)
summ

# %%
ablation

# %%
lead

# %% [markdown]
# ## G. How good must the text model be? Sensitivity sweep
# The confusion matrix is moved step by step from a perfect classifier (identity) to Model A, and from Model A
# to a random classifier (uniform). Each point retrains XGBoost (seed 42). The sweep is run twice: with the
# screening questions answered, and without them (free text only).

# %%
U5 = np.full((5, 5), 0.2)
sweep_cms = [(f'perfect→A {a:.2f}', (1 - a) * I5 + a * MODEL_A_CM) for a in [0, .25, .5, .75, 1.0]]
sweep_cms += [(f'A→random {b:.2f}', (1 - b) * MODEL_A_CM + b * U5) for b in [.25, .5, .75, 1.0]]
sw = []
for screen in ['yes', 'none']:
  for label, cm in sweep_cms:
    d, f1 = scenario_data(dict(text=cm, voice='simulated', screen=screen))
    tr, va = d[d.victim_id.isin(dev_tr_ids)], d[d.victim_id.isin(dev_va_ids)]
    te = d[d.victim_id.isin(test_ids)].reset_index(drop=True)
    r = dict(screening='answered' if screen == 'yes' else 'skipped', step=label, text_macro_F1=f1)
    for fname, feats in [('PSY', PSY_FEATS), ('PSY+LEGAL', FULL_FEATS)]:
        r[f'AUPRC_{fname}'] = average_precision_score(te.y, fit_xgb(tr, va, feats, 42).predict_proba(te[feats])[:, 1])
    r['legal_gain'] = r['AUPRC_PSY+LEGAL'] - r['AUPRC_PSY']
    sw.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
sweep = pd.DataFrame(sw).round(4); sweep.to_csv(f'{OUT}/text_quality_sweep.csv', index=False)
sweep

# %% [markdown]
# ## H. Figures

# %%
COL = {'XGBoost PSY': '#8a8f98', 'XGBoost PSY+LEGAL': '#2a6fdb'}
names = [s['name'] for s in SCENARIOS]
m = res[res.features.str.startswith('XGBoost')].groupby(['scenario', 'features']).AUPRC.agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(12, 4.8)); x = np.arange(len(names)); w = .38
for j, f in enumerate(['XGBoost PSY', 'XGBoost PSY+LEGAL']):
    r = m[m.features == f].set_index('scenario').reindex(names)
    ax.bar(x + (j - .5) * w, r['mean'], w, yerr=r['std'].fillna(0), capsize=3, color=COL[f], label=f.replace('XGBoost ', ''))
ax.set_xticks(x, [s.replace(' (', '\n(').replace(', ', ',\n').replace(': ', ':\n') for s in names], fontsize=7.5)
ax.axvline(N_WITH_SCREENING - .5, c='k', lw=.6, ls=':')
ax.text(N_WITH_SCREENING - .4, m['mean'].max() * 1.22, 'screening questions skipped →', fontsize=8)
ax.set_ylabel('Test AUPRC (mean ± SD over seeds)'); ax.set_ylim(0, m['mean'].max() * 1.3)
ax.set_title('Risk prediction with realistic text analysis (Model A → Model C)')
ax.legend(loc='upper right', ncol=2, frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_scenarios.png', dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for screening, ls in [('answered', '-'), ('skipped', '--')]:
    s = sweep[sweep.screening == screening].sort_values('text_macro_F1')
    ax.plot(s.text_macro_F1, s['AUPRC_PSY'], 'o' + ls, color='#8a8f98', ms=4, label=f'PSY (screening {screening})')
    ax.plot(s.text_macro_F1, s['AUPRC_PSY+LEGAL'], 'o' + ls, color='#2a6fdb', ms=4, label=f'PSY+LEGAL (screening {screening})')
a_f1 = sweep[sweep.step == 'perfect→A 1.00'].text_macro_F1.mean()
ax.axvline(a_f1, ls=':', c='#c0392b', lw=1.2)
ax.text(a_f1, ax.get_ylim()[1] * .97, f' Model A (macro-F1 {a_f1:.2f})', fontsize=8, color='#c0392b', va='top')
ax.set(xlabel='Text model macro-F1 on simulated replies', ylabel='Test AUPRC (XGBoost)',
       title='How good must the text model be?')
ax.legend(loc='lower right', fontsize=7.5); plt.tight_layout(); plt.savefig(f'{OUT}/fig_text_quality_sweep.png', dpi=200); plt.show()

# %% [markdown]
# ## I. Paper table and summary

# %%
paper = []
for sc in names:
    r = res[res.scenario == sc]
    row = {'Scenario': sc, 'Text macro-F1': f"{r.text_macro_F1.mean():.3f}" if r.text_macro_F1.notna().any() else '–'}
    for f, lab in [('DDS (rule-based)', 'DDS'), ('XGBoost PSY', 'PSY'), ('XGBoost PSY+LEGAL', 'PSY+LEGAL')]:
        q = r[r.features == f]
        row[f'AUPRC {lab}'] = f"{q.AUPRC.mean():.3f}" + (f" ± {q.AUPRC.std(ddof=1):.3f}" if len(q) > 1 else '')
    a = ablation[ablation.scenario == sc].iloc[0]
    row['Legal gain AUPRC (95% CI)'] = f"{a.delta_AUPRC:+.3f} ({a.AUPRC_ci_low:.3f} to {a.AUPRC_ci_high:.3f})"
    l = lead[(lead.scenario == sc) & (lead.features == 'PSY+LEGAL')].iloc[0]
    row['Episodes detected (PSY+LEGAL)'] = f"{l.detected_pct:.1f}%"
    paper.append(row)
paper = pd.DataFrame(paper); paper.to_csv(f'{OUT}/paper_table_integration.csv', index=False)
json.dump(dict(config=CFG, model_a_cm=MODEL_A_CM.round(4).tolist(), tone=TONE, model_b=MODEL_B,
               true_tone_prior=dict(zip(SENT, prior.round(4).tolist())),
               expected_model_a_macro_f1=expected_macro_f1(MODEL_A_CM),
               table=paper.to_dict('records'), sweep=sweep.to_dict('records')),
          open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print('Saved:', sorted(os.listdir(OUT)))
paper
