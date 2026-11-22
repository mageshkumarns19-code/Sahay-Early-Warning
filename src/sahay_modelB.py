# %% [markdown]
# # SAHAY – Model B: Multilingual Multi-Task Voice Model
# ### One shared XLS-R speech encoder, two task heads
#
# **Goal:** from a victim's voice in an IVRS call, estimate (1) depression-related vocal markers and (2) negative
# emotion (sadness, fear, anger), so the risk predictor (Model C) gets a voice-distress signal.
#
# | Head | Dataset | Labels | Speakers |
# |---|---|---|---|
# | `depression` | Androids Corpus (Tao, Esposito & Vinciarelli, Interspeech 2023) – reading + interview tasks | patient vs healthy control | ~116 |
# | `emotion` | EmoTa – Tamil emotional speech (CHiPSAL 2025) | anger, happiness, sadness, fear, neutral | 22 |
#
# **Method.** A frozen multilingual speech encoder (wav2vec2 XLS-R 300M, 128 languages) turns every 8-second chunk
# into a 1024-d embedding (mean of middle layers). A small network with a **shared layer and two heads** is trained on
# top (multi-task), compared with the same network trained per task (single-task) and an eGeMAPS + logistic-regression
# baseline. Freezing the encoder suits these small datasets (about 116 and 22 speakers) and keeps the run short.
#
# **Evaluation.** Speaker-independent 5-fold cross-validation (no speaker in both train and test), 3 seeds.
# Depression is scored per speaker (average of chunk probabilities), emotion per utterance.
# The notebook writes `model_b_for_integration.json` (sensitivity, specificity) for the A + B → C integration notebook.
#
# **Kaggle settings:** GPU T4 x2 (one GPU is used) · Internet **ON** · Input: your private `androids-corpus` dataset ·
# Secret: `HF_TOKEN` (Add-ons → Secrets) for EmoTa. Expected time: about 30–60 minutes.

# %%
import os, re, glob, json, time, random, warnings, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score, recall_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})

IN_KAGGLE = os.path.exists('/kaggle/working')
WORK = '/kaggle/working' if IN_KAGGLE else '.'
OUT = f'{WORK}/outputs_modelB'; os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get('SAHAY_QUICK', '0') == '1'
TINY = os.environ.get('SAHAY_TINY', '0') == '1'      # offline test: tiny random encoder + local test data

CFG = dict(
    ENCODER     = 'facebook/wav2vec2-xls-r-300m',
    LAYERS      = (6, 15),        # average hidden states of layers 6..15 (paralinguistic information peaks mid-network)
    SR          = 16000,
    CHUNK_SEC   = 8.0,
    MAX_CHUNKS  = 12,             # per recording (reading/interview recordings are long)
    SEEDS       = [42] if QUICK else [42, 7, 2026],
    N_FOLDS     = 5,
    EPOCHS      = 30 if QUICK else 150,
    LR          = 1e-3,
    WEIGHT_DECAY= 1e-3,
    HIDDEN      = 256,
    DROPOUT     = 0.3,
    BATCH       = 64,
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE} | quick: {QUICK} | tiny: {TINY}')

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# %% [markdown]
# ## 1. Find the Androids Corpus in the attached input
# File names follow `nn_XGmm_t.wav`: nn = speaker number, X = P (patient) or C (control), G = gender, mm = age,
# t = education level. A speaker is identified by `nn_X`, the same across the reading and interview tasks.

# %%
if TINY:
    ROOT = os.environ.get('SAHAY_ANDROIDS', './androids_test')
else:
    cands = [p for p in glob.glob('/kaggle/input/**/Reading-Task', recursive=True)]
    assert cands, 'Androids Corpus not found: attach your private androids-corpus dataset (Add Input).'
    ROOT = os.path.dirname(cands[0])
print('Androids root:', ROOT)
print('Top level:', sorted(os.listdir(ROOT)))

PAT = re.compile(r'(\d+)_([PC])([MF])(\d\d)_([0-9X])', re.I)
rows = []
for task, sub in [('reading', 'Reading-Task'), ('interview', 'Interview-Task/audio_clip')]:
    for f in sorted(glob.glob(f'{ROOT}/{sub}/**/*.wav', recursive=True)):
        m = PAT.search(os.path.basename(f)) or PAT.search(f)
        if not m:
            continue
        num, cond, g, age, edu = m.groups()
        rows.append(dict(path=f, task=task, speaker=f'{int(num)}_{cond.upper()}', label=int(cond.upper() == 'P'),
                         gender=g.upper(), age=int(age)))
if not any(r['task'] == 'interview' for r in rows):     # fall back to full interview recordings if clips are absent
    for f in sorted(glob.glob(f'{ROOT}/Interview-Task/audio/**/*.wav', recursive=True)):
        m = PAT.search(os.path.basename(f))
        if m:
            num, cond, g, age, edu = m.groups()
            rows.append(dict(path=f, task='interview', speaker=f'{int(num)}_{cond.upper()}',
                             label=int(cond.upper() == 'P'), gender=g.upper(), age=int(age)))
andr = pd.DataFrame(rows)
assert len(andr), 'No wav files matched the nn_XGmm_t naming pattern.'
print(andr.groupby(['task', 'label']).agg(files=('path', 'size'), speakers=('speaker', 'nunique')))
print('Total speakers:', andr.speaker.nunique())
if os.path.exists(f'{ROOT}/fold-list.csv'):
    print('\nfold-list.csv (first lines, for reference):'); print(open(f'{ROOT}/fold-list.csv').read()[:400])

# %% [markdown]
# ## 2. EmoTa (Tamil emotional speech) from Hugging Face
# Uses the `HF_TOKEN` Kaggle secret (your account must have accepted the EmoTa terms). File names are
# `<speaker>_<sentence>_<emotion>.wav` with emotion = ang, hap, sad, fea, neu.

# %%
EMO_CODES = {'ang': 'anger', 'hap': 'happiness', 'sad': 'sadness', 'fea': 'fear', 'neu': 'neutral'}
EMO_LABELS = ['anger', 'happiness', 'sadness', 'fear', 'neutral']
NEGATIVE = {'anger', 'sadness', 'fear'}

if TINY:
    EMO_DIR = os.environ.get('SAHAY_EMOTA', './emota_test')
else:
    from huggingface_hub import snapshot_download
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        token = os.environ.get('HF_TOKEN')
    assert token, 'Add your Hugging Face token as a Kaggle secret named HF_TOKEN (Add-ons → Secrets).'
    EMO_DIR = snapshot_download('aaivu-labs/EmoTa', repo_type='dataset', token=token,
                                local_dir=f'{WORK}/emota', allow_patterns=['data/*.wav', 'data/*.csv', 'meta/*'])
    del token
rows = []
for f in sorted(glob.glob(f'{EMO_DIR}/**/*.wav', recursive=True)):
    m = re.match(r'(\d+)_(\d+)_([a-z]{3})', os.path.basename(f).lower())
    if m and m.group(3) in EMO_CODES:
        rows.append(dict(path=f, speaker=f'ta{int(m.group(1))}', emotion=EMO_CODES[m.group(3)]))
emo = pd.DataFrame(rows)
emo['label'] = emo.emotion.map({e: i for i, e in enumerate(EMO_LABELS)})
print(f'EmoTa: {len(emo)} utterances, {emo.speaker.nunique()} speakers')
print(emo.emotion.value_counts().to_dict())

if QUICK:   # smaller subsets for a smoke test
    keep = andr.groupby('label').speaker.apply(lambda s: list(s.unique()[:10])).sum()
    andr = andr[andr.speaker.isin(keep)].groupby(['speaker', 'task']).head(2).reset_index(drop=True)
    emo = emo.groupby('speaker').head(20).reset_index(drop=True)

# %% [markdown]
# ## 3. Speech embeddings (frozen XLS-R) and eGeMAPS baseline features

# %%
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Config

if TINY:
    enc = Wav2Vec2Model(Wav2Vec2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
                                       intermediate_size=128, conv_dim=(32,) * 7))
    LAYERS = (1, 3)
else:
    enc = Wav2Vec2Model.from_pretrained(CFG['ENCODER']); LAYERS = CFG['LAYERS']
enc = enc.to(DEVICE).eval()
if DEVICE == 'cuda': enc = enc.half()

def chunks_of(y):
    n = int(CFG['CHUNK_SEC'] * CFG['SR'])
    if len(y) < n:
        return [y]
    starts = list(range(0, len(y) - n + 1, n))[:CFG['MAX_CHUNKS']]
    return [y[s:s + n] for s in starts]

@torch.no_grad()
def embed(paths):
    out_emb, out_idx = [], []
    for i, p in enumerate(paths):
        y, _ = librosa.load(p, sr=CFG['SR'], mono=True)
        y = y / (np.abs(y).max() + 1e-8) if len(y) else np.zeros(CFG['SR'], np.float32)
        for c in chunks_of(y):
            x = torch.tensor((c - c.mean()) / (c.std() + 1e-7), dtype=torch.float32, device=DEVICE)[None]
            if DEVICE == 'cuda': x = x.half()
            hs = enc(x, output_hidden_states=True).hidden_states
            v = torch.stack(hs[LAYERS[0]:LAYERS[1] + 1]).mean(0).mean(1)[0].float().cpu().numpy()
            out_emb.append(v); out_idx.append(i)
        if (i + 1) % 200 == 0: print(f'  embedded {i + 1}/{len(paths)} files')
    return np.stack(out_emb), np.array(out_idx)

t0 = time.time()
A_emb, A_idx = embed(andr.path.tolist()); print(f'Androids: {len(A_emb)} chunks from {len(andr)} files ({time.time() - t0:.0f}s)')
t0 = time.time()
E_emb, E_idx = embed(emo.path.tolist()); print(f'EmoTa: {len(E_emb)} chunks from {len(emo)} files ({time.time() - t0:.0f}s)')
del enc
if DEVICE == 'cuda': torch.cuda.empty_cache()

try:
    import opensmile
except ImportError:                      # not preinstalled on Kaggle
    subprocess.run(['pip', 'install', '-q', 'opensmile'], check=True)
    import opensmile
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
def egemaps(paths):
    return np.stack([smile.process_file(p).values[0] for p in paths])
t0 = time.time()
A_geo, E_geo = egemaps(andr.path.tolist()), egemaps(emo.path.tolist())
print(f'eGeMAPS features: {A_geo.shape[1]} per file ({time.time() - t0:.0f}s)')
np.savez_compressed(f'{OUT}/features.npz', A_emb=A_emb, A_idx=A_idx, E_emb=E_emb, E_idx=E_idx, A_geo=A_geo, E_geo=E_geo)

# %% [markdown]
# ## 4. Speaker-independent folds

# %%
def speaker_folds(df, label_col, seed=0):
    spk = df.groupby('speaker')[label_col].agg(lambda s: s.mode()[0]).reset_index()
    sgk = StratifiedGroupKFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=seed)
    fold = {}
    for k, (_, te) in enumerate(sgk.split(spk, spk[label_col], spk.speaker)):
        for s in spk.speaker.iloc[te]: fold[s] = k
    return df.speaker.map(fold).values

andr['fold'] = speaker_folds(andr, 'label')
emo['fold'] = speaker_folds(emo, 'label')
print('Androids speakers per fold:', andr.groupby('fold').speaker.nunique().tolist())
print('EmoTa speakers per fold:', emo.groupby('fold').speaker.nunique().tolist())

# %% [markdown]
# ## 5. Multi-task network and training

# %%
class VoiceNet(nn.Module):
    def __init__(self, d, tasks):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(d), nn.Dropout(CFG['DROPOUT']), nn.Linear(d, CFG['HIDDEN']),
                                    nn.GELU(), nn.Dropout(CFG['DROPOUT']))
        self.heads = nn.ModuleDict({t: nn.Linear(CFG['HIDDEN'], n) for t, n in tasks.items()})
    def forward(self, x, t):
        return self.heads[t](self.shared(x))

def train_net(task_data, seed):
    """task_data: {task: (X, y)}; returns trained net and per-task standardisation."""
    set_seed(seed)
    d = next(iter(task_data.values()))[0].shape[1]
    net = VoiceNet(d, {t: int(y.max()) + 1 if t != 'depression' else 2 for t, (_, y) in task_data.items()}).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
    data, lossf = {}, {}
    for t, (X, y) in task_data.items():
        data[t] = (torch.tensor(X, dtype=torch.float32, device=DEVICE), torch.tensor(y, dtype=torch.long, device=DEVICE))
        cnt = np.bincount(y, minlength=2 if t == 'depression' else len(EMO_LABELS)).astype(float)
        w = cnt.sum() / (len(cnt) * np.maximum(cnt, 1))
        lossf[t] = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
    tasks = list(task_data)
    for ep in range(CFG['EPOCHS']):
        net.train()
        perms = {t: torch.randperm(len(data[t][1]), device=DEVICE) for t in tasks}
        n_steps = max(int(np.ceil(len(data[t][1]) / CFG['BATCH'])) for t in tasks)
        for s in range(n_steps):
            loss = 0
            for t in tasks:                       # every step updates the shared layer with both tasks
                idx = perms[t][(s * CFG['BATCH']) % len(perms[t]):][:CFG['BATCH']]
                if len(idx) == 0: continue
                X, y = data[t]
                loss = loss + lossf[t](net(X[idx], t), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def predict(net, X, t):
    net.eval()
    return torch.softmax(net(torch.tensor(X, dtype=torch.float32, device=DEVICE), t), 1).cpu().numpy()

def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd

# %% [markdown]
# ## 6. Cross-validation: multi-task vs single-task vs eGeMAPS baseline

# %%
def dep_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(macro_F1=f1_score(y, pred, average='macro'), accuracy=accuracy_score(y, pred),
                AUROC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))

def emo_metrics(y, P):
    pred = P.argmax(1)
    neg_true = np.isin(y, [EMO_LABELS.index(e) for e in NEGATIVE]).astype(int)
    neg_p = P[:, [EMO_LABELS.index(e) for e in NEGATIVE]].sum(1)
    neg_pred = (neg_p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(neg_true, neg_pred, labels=[0, 1]).ravel()
    return dict(macro_F1=f1_score(y, pred, average='macro'), accuracy=accuracy_score(y, pred),
                UAR=recall_score(y, pred, average='macro'),
                negative_sensitivity=tp / max(tp + fn, 1), negative_specificity=tn / max(tn + fp, 1))

def speaker_level(chunk_idx, chunk_p, frame):
    """Average chunk probabilities per file, then per speaker (and per task)."""
    f = pd.DataFrame({'file': chunk_idx, 'p': chunk_p}).groupby('file').p.mean()
    d = frame.iloc[f.index].assign(p=f.values)
    return d

A_y_chunk = andr.label.values[A_idx]; E_y_chunk = emo.label.values[E_idx]
A_fold_chunk = andr.fold.values[A_idx]; E_fold_chunk = emo.fold.values[E_idx]
results, dep_pred_store, emo_pred_store = [], {}, {}

for seed in CFG['SEEDS']:
    dep_p = {m: np.zeros(len(A_emb)) for m in ['Multi-task', 'Single-task']}
    emo_p = {m: np.zeros((len(E_emb), len(EMO_LABELS))) for m in ['Multi-task', 'Single-task']}
    for k in range(CFG['N_FOLDS']):
        a_tr, a_te = A_fold_chunk != k, A_fold_chunk == k
        e_tr, e_te = E_fold_chunk != k, E_fold_chunk == k
        Xa_tr, Xa_te = standardise(A_emb[a_tr], A_emb[a_te])
        Xe_tr, Xe_te = standardise(E_emb[e_tr], E_emb[e_te])
        net = train_net({'depression': (Xa_tr, A_y_chunk[a_tr]), 'emotion': (Xe_tr, E_y_chunk[e_tr])}, seed)
        dep_p['Multi-task'][a_te] = predict(net, Xa_te, 'depression')[:, 1]
        emo_p['Multi-task'][e_te] = predict(net, Xe_te, 'emotion')
        dep_p['Single-task'][a_te] = predict(train_net({'depression': (Xa_tr, A_y_chunk[a_tr])}, seed), Xa_te, 'depression')[:, 1]
        emo_p['Single-task'][e_te] = predict(train_net({'emotion': (Xe_tr, E_y_chunk[e_tr])}, seed), Xe_te, 'emotion')
    for m in ['Multi-task', 'Single-task']:
        files = speaker_level(A_idx, dep_p[m], andr)
        for scope, d in [('all tasks', files), ('reading', files[files.task == 'reading']), ('interview', files[files.task == 'interview'])]:
            spk = d.groupby('speaker').agg(y=('label', 'first'), p=('p', 'mean'))
            if len(spk) and spk.y.nunique() > 1:
                results.append(dict(task='depression', scope=scope, model=f'XLS-R {m}', seed=seed, **dep_metrics(spk.y.values, spk.p.values)))
        ef = pd.DataFrame(emo_p[m]).groupby(E_idx).mean()
        results.append(dict(task='emotion', scope='EmoTa', model=f'XLS-R {m}', seed=seed,
                            **emo_metrics(emo.label.values[ef.index], ef.values)))
        dep_pred_store[(m, seed)] = files; emo_pred_store[(m, seed)] = (ef.index.values, ef.values)
    print(f'seed {seed} done')

# eGeMAPS + logistic regression baseline (file level, same speaker folds)
dep_b = np.zeros(len(andr)); emo_b = np.zeros((len(emo), len(EMO_LABELS)))
for k in range(CFG['N_FOLDS']):
    tr, te = andr.fold.values != k, andr.fold.values == k
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced'))
    dep_b[te] = clf.fit(A_geo[tr], andr.label.values[tr]).predict_proba(A_geo[te])[:, 1]
    tr, te = emo.fold.values != k, emo.fold.values == k
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced'))
    emo_b[te] = clf.fit(E_geo[tr], emo.label.values[tr]).predict_proba(E_geo[te])
fb = andr.assign(p=dep_b)
for scope, d in [('all tasks', fb), ('reading', fb[fb.task == 'reading']), ('interview', fb[fb.task == 'interview'])]:
    spk = d.groupby('speaker').agg(y=('label', 'first'), p=('p', 'mean'))
    if len(spk) and spk.y.nunique() > 1:
        results.append(dict(task='depression', scope=scope, model='eGeMAPS + LR', seed=0, **dep_metrics(spk.y.values, spk.p.values)))
results.append(dict(task='emotion', scope='EmoTa', model='eGeMAPS + LR', seed=0, **emo_metrics(emo.label.values, emo_b)))

res = pd.DataFrame(results); res.to_csv(f'{OUT}/results_by_seed.csv', index=False)
num = [c for c in res.columns if c not in ['task', 'scope', 'model', 'seed']]
summary = res.groupby(['task', 'scope', 'model'])[num].agg(['mean', 'std']).round(3)
summary.to_csv(f'{OUT}/summary.csv')
summary

# %% [markdown]
# ## 7. Figures

# %%
s0 = CFG['SEEDS'][0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
files = dep_pred_store[('Multi-task', s0)]
spk = files.groupby('speaker').agg(y=('label', 'first'), p=('p', 'mean'))
cm = confusion_matrix(spk.y, (spk.p >= .5).astype(int), labels=[0, 1], normalize='true')
idx, P = emo_pred_store[('Multi-task', s0)]
cm2 = confusion_matrix(emo.label.values[idx], P.argmax(1), labels=range(len(EMO_LABELS)), normalize='true')
for ax, mat, labs, title in [(axes[0], cm, ['Control', 'Patient'], 'Depression – Androids (speaker level)'),
                             (axes[1], cm2, EMO_LABELS, 'Emotion – EmoTa (Tamil)')]:
    ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)), labs, rotation=30, ha='right'); ax.set_yticks(range(len(labs)), labs)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if mat[i, j] > .5 else 'black')
    ax.set(title=f'{title}\nmulti-task XLS-R, row-normalised', xlabel='Predicted', ylabel='True')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_confusion_matrices.png', dpi=200, bbox_inches='tight'); plt.show()

plot = res[res.scope.isin(['all tasks', 'EmoTa'])].groupby(['task', 'model']).macro_F1.agg(['mean', 'std']).reset_index()
models = ['eGeMAPS + LR', 'XLS-R Single-task', 'XLS-R Multi-task']
colors = {'eGeMAPS + LR': '#b9bec7', 'XLS-R Single-task': '#8a8f98', 'XLS-R Multi-task': '#2a6fdb'}
fig, ax = plt.subplots(figsize=(7, 4)); tasks = ['depression', 'emotion']; x = np.arange(2); w = .26
for j, m in enumerate(models):
    r = plot[plot.model == m].set_index('task').reindex(tasks)
    ax.bar(x + (j - 1) * w, r['mean'], w, yerr=r['std'].fillna(0), capsize=3, color=colors[m], label=m)
ax.set_xticks(x, ['Depression (Androids)', 'Emotion (EmoTa, Tamil)']); ax.set_ylabel('Macro-F1 (mean ± SD over seeds)')
ax.set_ylim(0, 1); ax.legend(loc='upper left', ncol=3, frameon=False, fontsize=8)
ax.set_title('Model B: multi-task vs single-task vs baseline')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_model_comparison.png', dpi=200); plt.show()

# %% [markdown]
# ## 8. Values for the A + B → C integration notebook
# The integration notebook needs the voice model's sensitivity and specificity for detecting distressed speech.
# We use the multi-task depression head at speaker level (mean over seeds). Paste these two numbers into `MODEL_B`
# in the integration notebook (section A).

# %%
dm = res[(res.task == 'depression') & (res.scope == 'all tasks') & (res.model == 'XLS-R Multi-task')]
em = res[(res.task == 'emotion') & (res.model == 'XLS-R Multi-task')]
integration = dict(sensitivity=round(float(dm.sensitivity.mean()), 3), specificity=round(float(dm.specificity.mean()), 3),
                   source='Model B multi-task depression head, Androids Corpus, speaker level, mean over seeds',
                   negative_emotion_sensitivity=round(float(em.negative_sensitivity.mean()), 3),
                   negative_emotion_specificity=round(float(em.negative_specificity.mean()), 3))
json.dump(integration, open(f'{OUT}/model_b_for_integration.json', 'w'), indent=1)
print('Paste into the integration notebook:  MODEL_B = dict(sensitivity=%.3f, specificity=%.3f)'
      % (integration['sensitivity'], integration['specificity']))

def fmt(df):
    out = []
    for (t, sc, m), d in df.groupby(['task', 'scope', 'model']):
        cols = ['macro_F1', 'accuracy', 'AUROC', 'sensitivity', 'specificity'] if t == 'depression' else \
               ['macro_F1', 'accuracy', 'UAR', 'negative_sensitivity', 'negative_specificity']
        out.append({'Task': t, 'Scope': sc, 'Model': m,
                    **{c: f"{d[c].mean():.3f} ± {d[c].std(ddof=1) if len(d) > 1 else 0:.3f}" for c in cols if c in d}})
    return pd.DataFrame(out)
paper = fmt(res); paper.to_csv(f'{OUT}/paper_table_modelB.csv', index=False)
json.dump(dict(config=CFG, n_androids_files=len(andr), n_androids_speakers=int(andr.speaker.nunique()),
               n_emota=len(emo), integration=integration, results=paper.to_dict('records')),
          open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print('Saved:', sorted(os.listdir(OUT)))
paper
