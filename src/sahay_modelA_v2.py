# %% [markdown]
# # SAHAY – Model A: Multilingual Multi-Task Text Model
# ### One shared MuRIL encoder, several task heads
#
# **Goal:** read a victim's check-in text and output (1) its emotional tone and (2) the likely *cause* of distress,
# so the system can route the right intervention. Optional heads (emotion intensity, suicide risk) switch on
# automatically when those restricted datasets are attached.
#
# | Head | Dataset | Labels | Status |
# |---|---|---|---|
# | `sentiment` | DravidianCodeMix – Tamil-English + Malayalam-English (Chakravarthi et al., 2022) | Positive, Negative, Mixed_feelings, unknown_state, not_native | open, downloaded automatically |
# | `cause` | CAMS – causes of mental-health distress (Garg et al., LREC 2022) | no_reason, bias_or_abuse, jobs_career, medication, relationship, alienation | open, downloaded automatically |
# | `emotion` *(optional)* | EmoInHindi (Singh et al., LREC 2022) | 16 emotions, multi-label | attach your private copy |
# | `risk` *(optional)* | UMD Reddit Suicidality / CLPsych 2019 | no / low / moderate / severe | attach your private copy |
#
# **Experiments**
# 1. Multi-task model (all heads share one encoder) vs single-task models (one encoder per task), 3 seeds.
# 2. TF-IDF + Logistic Regression baseline.
# 3. Per-language results (Tamil vs Malayalam), confusion matrices, paired bootstrap test of multi-task vs single-task.
# 4. Cause → intervention mapping (links predictions to PoA-Act relief, protection, livelihood and counselling measures).
#
# **Kaggle settings:** Accelerator = **GPU T4 x2**, **Internet = ON** (to download MuRIL and the two open datasets).
# For the full run use *Save Version → Save & Run All (Commit)*. Expected time: about 5–7 hours.
# A time guard stops starting new runs after 10.5 hours, and results are saved after every run, so the
# 12-hour Kaggle limit can never wipe out the results.

# %%
import os, re, json, time, math, glob, random, warnings, subprocess, zipfile, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})

IN_KAGGLE = os.path.exists('/kaggle/working')
WORK = '/kaggle/working' if IN_KAGGLE else '.'
OUT = f'{WORK}/outputs_modelA'; DATA = f'{WORK}/data'
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

QUICK = os.environ.get('SAHAY_QUICK', '0') == '1'   # small smoke test
TINY = os.environ.get('SAHAY_TINY', '0') == '1'     # offline test with a tiny random model (no MuRIL download)

CFG = dict(
    MODEL_NAME   = 'google/muril-base-cased',   # alternatives: 'xlm-roberta-base', 'ai4bharat/indic-bert'
    MAX_LEN      = {'sentiment': 128, 'cause': 256, 'emotion': 128, 'risk': 256},
    BATCH        = 32,            # 16 per GPU on T4 x2
    LR           = 3e-5,          # encoder (scaled up with the larger batch)
    HEAD_LR      = 1e-4,          # task heads
    WEIGHT_DECAY = 0.01,
    WARMUP       = 0.10,
    EPOCHS       = 1 if QUICK else 4,
    TIME_BUDGET_HOURS = 10.5,     # stop starting new runs after this, so Kaggle's 12-hour limit is never hit
    PATIENCE     = 2,             # early stopping on mean validation macro-F1
    SEEDS        = [42] if QUICK else [42, 7, 2026],
    SAMPLE_TEMP  = 0.5,           # multi-task sampling: p(task) proportional to n_task ** 0.5
    RUN_SINGLE_TASK = True,       # also train one model per task (needed for the multi-task comparison)
    USE_ALL_GPUS = True,          # use both T4s via DataParallel
    FP16         = True,
    N_BOOT       = 200 if QUICK else 1000,
    SUBSAMPLE    = 1500 if QUICK else None,   # rows per split in quick mode
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_GPU = torch.cuda.device_count()
print(f'Device: {DEVICE} | GPUs: {N_GPU} | quick: {QUICK} | tiny: {TINY}')

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# %% [markdown]
# ## 1. Optional restricted datasets
# Upload your licensed copies as **private** Kaggle datasets, attach them, and fill in the path and column names.
# Leave `path=None` to skip a head. Each CSV needs a text column and a label column; if it has no split column,
# an 80/10/10 split is made. For multi-label data, labels are separated by `label_sep` (e.g. "fear,sadness").

# %%
OPTIONAL = {
    'emotion': dict(path=None,            # e.g. '/kaggle/input/emoinhindi-private/emoinhindi.csv'
                    text_col='utterance', label_col='emotion', split_col=None,
                    multilabel=True, label_sep=','),
    'risk':    dict(path=None,            # e.g. '/kaggle/input/umd-suicidality-private/posts_with_labels.csv'
                    text_col='post_body', label_col='raw_label', split_col=None,
                    multilabel=False, label_sep=None),
}

# %% [markdown]
# ## 2. Download and parse the open datasets

# %%
def git_clone(url, dest):
    if not os.path.exists(dest):
        subprocess.run(['git', 'clone', '-q', '--depth', '1', url, dest], check=True)

git_clone('https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset', f'{DATA}/dcm_repo')
git_clone('https://github.com/drmuskangarg/CAMS', f'{DATA}/cams_repo')
if not os.path.exists(f'{DATA}/dcm'):
    zipfile.ZipFile(f'{DATA}/dcm_repo/DravidianCodeMix-2020.zip').extractall(f'{DATA}/dcm')
DCM_DIR = glob.glob(f'{DATA}/dcm/**/tamil_sentiment_full_train.csv', recursive=True)[0].rsplit('/', 1)[0]

SENT_LABELS = ['Positive', 'Negative', 'Mixed_feelings', 'unknown_state', 'not_native']
_pat = re.compile(r'^(.*\S)\s*[\t;]\s*(Positive|Negative|Mixed_feelings|unknown_state|not-Tamil|not-malayalam)\s*[;\t]*\s*$')

def parse_dcm(path):
    """The files mix tab and semicolon separators; parse each line with a regex anchored on the label."""
    rows, buf = [], ''
    for line in open(path, encoding='utf-8', errors='replace'):
        line = buf + line.rstrip('\r\n'); m = _pat.match(line)
        if m: rows.append(m.groups()); buf = ''
        else: buf = line + ' '
    d = pd.DataFrame(rows, columns=['text', 'label'])
    d['label'] = d.label.replace({'not-Tamil': 'not_native', 'not-malayalam': 'not_native'})
    return d

sent = {}
for split, sfx in [('train', 'train'), ('val', 'dev'), ('test', 'test')]:
    parts = []
    for lang, stem in [('Tamil', 'tamil_sentiment_full'), ('Malayalam', 'mal_full_sentiment')]:
        d = parse_dcm(f'{DCM_DIR}/{stem}_{sfx}.csv'); d['lang'] = lang; parts.append(d)
    sent[split] = pd.concat(parts, ignore_index=True)

CAUSE_LABELS = ['no_reason', 'bias_or_abuse', 'jobs_career', 'medication', 'relationship', 'alienation']  # CAMS codes 0-5
cdir = f'{DATA}/cams_repo/CAMS/data'
c_tr = pd.read_csv(f'{cdir}/IntentSDCNL_Training.csv').rename(columns={'selftext': 'text', 'ANNOTATIONS': 'y'})
c_add = pd.read_csv(f'{cdir}/added_CAMS_data.csv', encoding='latin-1').rename(columns={'selftext': 'text', 'cause': 'y'})
c_te = pd.read_csv(f'{cdir}/IntentSDCNL_Testing.csv').rename(columns={'selftext': 'text', 'ANNOTATIONS': 'y'})
def clean_cams(d):
    d = d.dropna(subset=['text', 'y']).copy(); d['y'] = d.y.astype(int)
    d = d[d.y.between(0, 5)]; d['label'] = d.y.map(dict(enumerate(CAUSE_LABELS)))
    d['text'] = d.text.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    return d[['text', 'label']].reset_index(drop=True)
c_train_all = pd.concat([clean_cams(c_tr), clean_cams(c_add)], ignore_index=True)
c_train, c_val = train_test_split(c_train_all, test_size=0.10, stratify=c_train_all.label, random_state=42)
cause = dict(train=c_train.reset_index(drop=True), val=c_val.reset_index(drop=True), test=clean_cams(c_te))

TASKS = {
    'sentiment': dict(data=sent, labels=SENT_LABELS, multilabel=False),
    'cause': dict(data=cause, labels=CAUSE_LABELS, multilabel=False),
}

for name, spec in OPTIONAL.items():
    if spec['path'] and os.path.exists(spec['path']):
        d = pd.read_csv(spec['path']).rename(columns={spec['text_col']: 'text', spec['label_col']: 'label'})
        d = d.dropna(subset=['text', 'label'])
        if spec['multilabel']:
            d['label'] = d.label.astype(str).str.split(spec['label_sep']).apply(lambda xs: [x.strip() for x in xs if x.strip()])
            labels = sorted({x for xs in d.label for x in xs})
        else:
            d['label'] = d.label.astype(str); labels = sorted(d.label.unique())
        if spec['split_col']:
            parts = {k: d[d[spec['split_col']] == k].reset_index(drop=True) for k in ['train', 'val', 'test']}
        else:
            tr, rest = train_test_split(d, test_size=.2, random_state=42)
            va, te = train_test_split(rest, test_size=.5, random_state=42)
            parts = dict(train=tr.reset_index(drop=True), val=va.reset_index(drop=True), test=te.reset_index(drop=True))
        TASKS[name] = dict(data=parts, labels=labels, multilabel=spec['multilabel'])
        print(f'Optional head "{name}" enabled with {len(labels)} labels')

if CFG['SUBSAMPLE']:
    for t in TASKS.values():
        for k, d in t['data'].items():
            t['data'][k] = d.sample(min(len(d), CFG['SUBSAMPLE']), random_state=0).reset_index(drop=True)

rows = []
for tname, t in TASKS.items():
    for k, d in t['data'].items():
        rows.append(dict(task=tname, split=k, n=len(d)))
print(pd.DataFrame(rows).pivot(index='task', columns='split', values='n')[['train', 'val', 'test']])
for tname, t in TASKS.items():
    if not t['multilabel']:
        print(f'\n{tname} train label counts:', t['data']['train'].label.value_counts().to_dict())

# %% [markdown]
# ## 3. Baseline: TF-IDF (character n-grams) + Logistic Regression
# Character n-grams handle code-mixed and transliterated text well, so this is a strong, cheap baseline.

# %%
def task_metrics(y_true, y_pred, labels, multilabel=False):
    if multilabel:
        return dict(micro_F1=f1_score(y_true, y_pred, average='micro', zero_division=0),
                    macro_F1=f1_score(y_true, y_pred, average='macro', zero_division=0))
    return dict(accuracy=accuracy_score(y_true, y_pred),
                macro_F1=f1_score(y_true, y_pred, average='macro', labels=list(range(len(labels))), zero_division=0),
                weighted_F1=f1_score(y_true, y_pred, average='weighted', zero_division=0))

def enc_labels(d, t):
    if t['multilabel']:
        idx = {l: i for i, l in enumerate(t['labels'])}
        Y = np.zeros((len(d), len(t['labels'])), np.float32)
        for r, xs in enumerate(d.label):
            for x in xs:
                if x in idx: Y[r, idx[x]] = 1
        return Y
    return d.label.map({l: i for i, l in enumerate(t['labels'])}).values

results = []
for tname, t in TASKS.items():
    if t['multilabel']:
        continue
    tr, te = t['data']['train'], t['data']['test']
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, max_features=200000, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(tr.text), vec.transform(te.text)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=4.0).fit(Xtr, enc_labels(tr, t))
    m = task_metrics(enc_labels(te, t), clf.predict(Xte), t['labels'])
    results.append(dict(model='TF-IDF + LR', task=tname, seed=0, **m))
    print(tname, {k: round(v, 4) for k, v in m.items()})

# %% [markdown]
# ## 4. Tokeniser and encoder

# %%
if TINY:
    # Offline test mode only: build a small WordPiece tokenizer and a random 2-layer BERT.
    from tokenizers import BertWordPieceTokenizer
    from transformers import PreTrainedTokenizerFast, BertConfig, BertModel
    corpus = pd.concat([t['data']['train'].text for t in TASKS.values()]).astype(str).tolist()
    wp = BertWordPieceTokenizer(lowercase=False); wp.train_from_iterator(corpus, vocab_size=8000)
    os.makedirs(f'{WORK}/tiny_tok', exist_ok=True); wp.save(f'{WORK}/tiny_tok/tokenizer.json')
    tok = PreTrainedTokenizerFast(tokenizer_file=f'{WORK}/tiny_tok/tokenizer.json', unk_token='[UNK]',
                                  pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
    def load_encoder():
        return BertModel(BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2,
                                    num_attention_heads=2, intermediate_size=128))
else:
    tok = AutoTokenizer.from_pretrained(CFG['MODEL_NAME'])
    def load_encoder():
        return AutoModel.from_pretrained(CFG['MODEL_NAME'])

for tname, t in TASKS.items():
    for k, d in t['data'].items():
        t.setdefault('ids', {})[k] = tok(d.text.astype(str).tolist(), truncation=True,
                                        max_length=CFG['MAX_LEN'].get(tname, 128))['input_ids']
        t.setdefault('y', {})[k] = enc_labels(d, t)
lens = {n: int(np.median([len(x) for x in t['ids']['train']])) for n, t in TASKS.items()}
print('Median token length per task:', lens)

# %% [markdown]
# ## 5. Multi-task model and training loop
# Each training step samples one task (probability proportional to its size ** 0.5, so the small CAMS task is
# up-sampled), takes a batch from it and updates the shared encoder plus that task's head.
# Multi-class heads use class-weighted cross-entropy; multi-label heads use binary cross-entropy.

# %%
class MultiTaskModel(nn.Module):
    def __init__(self, encoder, task_sizes):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        h = encoder.config.hidden_size
        self.heads = nn.ModuleDict({t: nn.Linear(h, n) for t, n in task_sizes.items()})
    def forward(self, input_ids, attention_mask, task):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.heads[task](self.drop(cls))

def collate(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s)
    return ids, (ids != tok.pad_token_id).long()

def predict(net, tname, split, bs=64):
    net.eval(); ids = TASKS[tname]['ids'][split]; out = []
    order = np.argsort([len(x) for x in ids])          # length-sorted batches are much faster
    with torch.no_grad(), torch.autocast('cuda', enabled=CFG['FP16'] and DEVICE == 'cuda'):
        for i in range(0, len(ids), bs):
            b = order[i:i + bs]; x, m = collate([ids[j] for j in b])
            out.append(net(x.to(DEVICE), m.to(DEVICE), tname).float().cpu())
    logits = torch.cat(out).numpy(); res = np.empty_like(logits); res[order] = logits
    return res

def decide(tname, logits):
    if TASKS[tname]['multilabel']:
        return (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
    return logits.argmax(1)

def train_run(task_names, seed, tag):
    set_seed(seed)
    core = MultiTaskModel(load_encoder(), {t: len(TASKS[t]['labels']) for t in task_names}).to(DEVICE)
    net = nn.DataParallel(core) if (CFG['USE_ALL_GPUS'] and N_GPU > 1) else core
    sizes = np.array([len(TASKS[t]['ids']['train']) for t in task_names], float)
    probs = sizes ** CFG['SAMPLE_TEMP']; probs /= probs.sum()
    steps_per_epoch = int(math.ceil(sizes.sum() / CFG['BATCH']))
    total = steps_per_epoch * CFG['EPOCHS']
    params = [{'params': core.encoder.parameters(), 'lr': CFG['LR']},
              {'params': core.heads.parameters(), 'lr': CFG['HEAD_LR']}]
    opt = torch.optim.AdamW(params, weight_decay=CFG['WEIGHT_DECAY'])
    sched = get_linear_schedule_with_warmup(opt, int(CFG['WARMUP'] * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG['FP16'] and DEVICE == 'cuda')
    lossf = {}
    for t in task_names:
        if TASKS[t]['multilabel']:
            lossf[t] = nn.BCEWithLogitsLoss()
        else:
            cnt = np.bincount(TASKS[t]['y']['train'], minlength=len(TASKS[t]['labels'])).astype(float)
            w = np.sqrt(cnt.sum() / (len(cnt) * np.maximum(cnt, 1))); w /= w.mean()
            lossf[t] = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    perms = {t: np.random.permutation(len(TASKS[t]['ids']['train'])) for t in task_names}
    ptr = {t: 0 for t in task_names}
    best, best_state, wait, history = -1, None, 0, []
    for ep in range(CFG['EPOCHS']):
        net.train(); t0 = time.time(); run_loss = 0
        for step in range(steps_per_epoch):
            t = task_names[np.random.choice(len(task_names), p=probs)]
            if ptr[t] + CFG['BATCH'] > len(perms[t]):
                perms[t] = np.random.permutation(len(perms[t])); ptr[t] = 0
            b = perms[t][ptr[t]:ptr[t] + CFG['BATCH']]; ptr[t] += CFG['BATCH']
            x, m = collate([TASKS[t]['ids']['train'][j] for j in b])
            y = torch.tensor(TASKS[t]['y']['train'][b], device=DEVICE)
            with torch.autocast('cuda', enabled=CFG['FP16'] and DEVICE == 'cuda'):
                logits = net(x.to(DEVICE), m.to(DEVICE), t)
            loss = lossf[t](logits.float(), y.float() if TASKS[t]['multilabel'] else y.long())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(core.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item()
        val = {t: task_metrics(TASKS[t]['y']['val'], decide(t, predict(net, t, 'val')), TASKS[t]['labels'],
                               TASKS[t]['multilabel'])['macro_F1'] for t in task_names}
        score = float(np.mean(list(val.values())))
        history.append(dict(run=tag, seed=seed, epoch=ep + 1, train_loss=run_loss / steps_per_epoch,
                            val_mean_macro_F1=score, **{f'val_{k}': v for k, v in val.items()}))
        print(f'  [{tag} seed {seed}] epoch {ep + 1}: loss {run_loss / steps_per_epoch:.4f} | val macro-F1 '
              + ' '.join(f'{k}={v:.4f}' for k, v in val.items()) + f' | {time.time() - t0:.0f}s')
        if score > best:
            best, wait = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG['PATIENCE']:
                break
    core.load_state_dict(best_state)
    test_logits = {t: predict(net, t, 'test') for t in task_names}
    return core, test_logits, history

# %% [markdown]
# ## 6. Run experiments: multi-task vs single-task, 3 seeds

# %%
ALL = list(TASKS)
history, test_preds = [], {}
T_START = time.time(); run_minutes = []
for seed in CFG['SEEDS']:
    runs = [('Multi-task', ALL)]
    if CFG['RUN_SINGLE_TASK'] and len(ALL) > 1:
        runs += [(f'Single-task', [t]) for t in ALL]
    for tag, tl in runs:
        elapsed_h = (time.time() - T_START) / 3600
        est_h = (max(run_minutes) / 60) if run_minutes else 0
        if elapsed_h + est_h > CFG['TIME_BUDGET_HOURS']:
            print(f'Time budget reached ({elapsed_h:.1f} h used): skipping {tag} {tl} seed {seed}')
            continue
        t0 = time.time()
        core, logits, hist = train_run(tl, seed, tag)
        history += hist
        for t in tl:
            pred = decide(t, logits[t])
            m = task_metrics(TASKS[t]['y']['test'], pred, TASKS[t]['labels'], TASKS[t]['multilabel'])
            results.append(dict(model=f'MuRIL {tag}', task=t, seed=seed, **m))
            test_preds[(tag, t, seed)] = (logits[t], pred)
        run_minutes.append((time.time() - t0) / 60)
        print(f'{tag} {tl} seed {seed} done in {run_minutes[-1]:.1f} min (total {(time.time() - T_START) / 3600:.2f} h)')
        pd.DataFrame(results).to_csv(f'{OUT}/results_by_run.csv', index=False)      # saved after every run
        pd.DataFrame(history).to_csv(f'{OUT}/training_history.csv', index=False)
        if tag == 'Multi-task' and seed == CFG['SEEDS'][0]:
            os.makedirs(f'{OUT}/model_A', exist_ok=True)
            torch.save({k: v.half() for k, v in core.state_dict().items()}, f'{OUT}/model_A/model_A_fp16.pt')
            tok.save_pretrained(f'{OUT}/model_A/tokenizer')
            json.dump({'model_name': CFG['MODEL_NAME'], 'tasks': {t: TASKS[t]['labels'] for t in ALL},
                       'multilabel': {t: TASKS[t]['multilabel'] for t in ALL}, 'max_len': CFG['MAX_LEN']},
                      open(f'{OUT}/model_A/config.json', 'w'), indent=1)
        del core
        if DEVICE == 'cuda': torch.cuda.empty_cache()

res = pd.DataFrame(results); res.to_csv(f'{OUT}/results_by_run.csv', index=False)
pd.DataFrame(history).to_csv(f'{OUT}/training_history.csv', index=False)
metric_cols = [c for c in ['accuracy', 'macro_F1', 'weighted_F1', 'micro_F1'] if c in res]
summary = res.groupby(['task', 'model'])[metric_cols].agg(['mean', 'std']).round(4)
summary.to_csv(f'{OUT}/summary.csv')
summary

# %% [markdown]
# ## 7. Is multi-task better? Paired bootstrap on the test set (seed 42)

# %%
def boot_f1(y, p1, p2, n_boot, seed=0, multilabel=False):
    rng = np.random.default_rng(seed); n = len(y); d = []
    avg = 'macro'
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(f1_score(y[i], p2[i], average=avg, zero_division=0) - f1_score(y[i], p1[i], average=avg, zero_division=0))
    d = np.array(d)
    return dict(delta_macro_F1=d.mean(), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                p_value=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

s0 = CFG['SEEDS'][0]; comp = []
if CFG['RUN_SINGLE_TASK'] and len(ALL) > 1:
    for t in ALL:
        if ('Single-task', t, s0) not in test_preds:
            continue
        y = TASKS[t]['y']['test']
        comp.append(dict(task=t, comparison='Multi-task minus Single-task',
                         **boot_f1(y, test_preds[('Single-task', t, s0)][1], test_preds[('Multi-task', t, s0)][1],
                                   CFG['N_BOOT'], multilabel=TASKS[t]['multilabel'])))
comp = pd.DataFrame(comp).round(4); comp.to_csv(f'{OUT}/multitask_vs_singletask_bootstrap.csv', index=False)
comp

# %% [markdown]
# ## 8. Per-language results and per-class F1 (multi-task model, seed 42)

# %%
pred_s = test_preds[('Multi-task', 'sentiment', s0)][1]
st = TASKS['sentiment']['data']['test']; ys = TASKS['sentiment']['y']['test']
lang_rows = []
for lang in st.lang.unique():
    i = (st.lang == lang).values
    lang_rows.append(dict(language=lang, n=int(i.sum()), **task_metrics(ys[i], pred_s[i], SENT_LABELS)))
by_lang = pd.DataFrame(lang_rows).round(4); by_lang.to_csv(f'{OUT}/sentiment_by_language.csv', index=False)
print(by_lang.to_string(index=False))

reports = []
for t in ALL:
    if TASKS[t]['multilabel']:
        continue
    rep = classification_report(TASKS[t]['y']['test'], test_preds[('Multi-task', t, s0)][1],
                                labels=list(range(len(TASKS[t]['labels']))), target_names=TASKS[t]['labels'],
                                output_dict=True, zero_division=0)
    for lab in TASKS[t]['labels']:
        reports.append(dict(task=t, label=lab, **{k: round(v, 4) for k, v in rep[lab].items()}))
per_class = pd.DataFrame(reports); per_class.to_csv(f'{OUT}/per_class_report.csv', index=False)
per_class

# %% [markdown]
# ## 9. Figures

# %%
single = [t for t in ALL if not TASKS[t]['multilabel']]
fig, axes = plt.subplots(1, len(single), figsize=(6 * len(single), 5))
axes = np.atleast_1d(axes)
for ax, t in zip(axes, single):
    cm = confusion_matrix(TASKS[t]['y']['test'], test_preds[('Multi-task', t, s0)][1],
                          labels=list(range(len(TASKS[t]['labels']))), normalize='true')
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)), TASKS[t]['labels'], rotation=40, ha='right')
    ax.set_yticks(range(len(cm)), TASKS[t]['labels'])
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > .5 else 'black')
    ax.set(title=f'{t} – multi-task MuRIL (row-normalised)', xlabel='Predicted', ylabel='True')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_confusion_matrices.png', dpi=200, bbox_inches='tight'); plt.show()

plot = res[res.task.isin(single)].groupby(['task', 'model']).macro_F1.agg(['mean', 'std']).reset_index()
models = [m for m in ['TF-IDF + LR', 'MuRIL Single-task', 'MuRIL Multi-task'] if m in plot.model.unique()]
colors = {'TF-IDF + LR': '#b9bec7', 'MuRIL Single-task': '#8a8f98', 'MuRIL Multi-task': '#2a6fdb'}
fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(single)); w = .8 / len(models)
for j, mname in enumerate(models):
    r = plot[plot.model == mname].set_index('task').reindex(single)
    ax.bar(x + (j - (len(models) - 1) / 2) * w, r['mean'], w, yerr=r['std'].fillna(0), capsize=3,
           color=colors[mname], label=mname)
ax.set_xticks(x, single); ax.set_ylabel('Test macro-F1 (mean ± SD over seeds)')
ax.set_ylim(0, plot['mean'].max() * 1.3); ax.legend(loc='upper left', ncol=3, frameon=False, fontsize=8)
ax.set_title('Model A: multi-task vs single-task vs baseline')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_model_comparison.png', dpi=200); plt.show()

# %% [markdown]
# ## 10. From predicted cause to intervention
# The `cause` head links each check-in to the agency that can act on it. This mapping is a proposal for the
# framework and should be reviewed with district officials and counsellors.

# %%
CAUSE_TO_INTERVENTION = {
    'bias_or_abuse': 'Protection: inform SP / protection officer; witness-protection review; legal aid (DLSA)',
    'jobs_career':   'Livelihood: release pending relief; employment / rehabilitation scheme (DSWO, Collector)',
    'medication':    'Medical: district hospital or psychiatric follow-up (DMHP)',
    'relationship':  'Psychosocial: counselling; family and community support (NGO linkage)',
    'alienation':    'Social inclusion: community support, vigilance & monitoring committee, peer support',
    'no_reason':     'Counselling check-in; continue monitoring',
}
ex = TASKS['cause']['data']['test'].copy()
lg = test_preds[('Multi-task', 'cause', s0)][0]
p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
ex['predicted_cause'] = [CAUSE_LABELS[i] for i in p.argmax(1)]
ex['confidence'] = p.max(1).round(3)
ex['recommended_intervention'] = ex.predicted_cause.map(CAUSE_TO_INTERVENTION)
ex['text'] = ex.text.str.slice(0, 120) + '…'
ex[['text', 'label', 'predicted_cause', 'confidence', 'recommended_intervention']].head(8).to_csv(
    f'{OUT}/example_cause_to_intervention.csv', index=False)
ex[['label', 'predicted_cause', 'confidence', 'recommended_intervention']].head(8)

# %% [markdown]
# ## 11. Paper table and saved model
# The saved model (`model_A/`) is used later to feed text-distress scores into Model C (the dynamic risk predictor).

# %%
def fmt(df):
    out = []
    for (t, mname), d in df.groupby(['task', 'model']):
        out.append({'Task': t, 'Model': mname,
                    **{c: f"{d[c].mean():.3f} ± {d[c].std(ddof=1) if len(d) > 1 else 0:.3f}" for c in metric_cols if d[c].notna().any()}})
    return pd.DataFrame(out)
paper = fmt(res); paper.to_csv(f'{OUT}/paper_table_modelA.csv', index=False)
json.dump(dict(config={k: v for k, v in CFG.items()}, tasks={t: TASKS[t]['labels'] for t in ALL},
               n={t: {k: len(v) for k, v in TASKS[t]['data'].items()} for t in ALL},
               results=paper.to_dict('records'), multitask_vs_single=comp.to_dict('records'),
               by_language=by_lang.to_dict('records')),
          open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print('Saved:', sorted(os.listdir(OUT)))
paper
