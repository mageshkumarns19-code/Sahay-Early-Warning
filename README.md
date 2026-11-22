# Legal-process-aware early warning of psychological distress in victims of atrocity crimes

Code, notebooks and results for the article:

> **Legal-Process-Aware Early Warning of Psychological Distress in Victims of Atrocity Crimes: A Proof-of-Concept Multilingual Artificial Intelligence Framework** (manuscript submitted to the *International Journal of Law and Psychiatry*).

The framework (working name **SAHAY**) combines three models:

| Model | What it does | Data |
|---|---|---|
| **A – text** | Multi-task MuRIL model: sentiment (5 classes) and cause of distress (6 classes) from short check-ins | DravidianCodeMix (Tamil/Malayalam code-mixed), CAMS |
| **B – voice** | Frozen XLS-R 300M encoder + multi-task head: depression and emotion from speech | Androids Corpus (Italian), EmoTa (Tamil) |
| **C – risk predictor** | Logistic regression, XGBoost and Attention-LSTM: risk of distress escalation in the next 14 days, with SHAP explanations | Simulated cohort of 6,000 victims × 104 weeks, grounded in NCRB statistics and the PoA Rules |

The integration notebook carries the measured error rates of Models A and B into Model C.

> **Important:** no real victim data were used. Model C is trained and tested on a simulated cohort, so its results show what the framework *can* detect, not how strong legal-process effects are in real victims. See Section 6.5 of the article.

---

## Repository structure

```
sahay-early-warning/
├── notebooks/                      # Kaggle notebooks used for the reported results
│   ├── 01_ModelC_risk_predictor.ipynb
│   ├── 02_ModelA_text.ipynb                    # main Model A run (batch 16, LR 2e-5, ≤6 epochs)
│   ├── 02b_ModelA_text_v2_robustness.ipynb     # robustness run (batch 32, LR 3e-5, ≤4 epochs)
│   ├── 03_ModelB_voice.ipynb
│   └── 04_Integration_A_B_to_C.ipynb
├── src/                            # the same code as plain Python scripts (# %% cells)
├── results/                        # outputs of the reported Kaggle runs (CSV, JSON, PNG)
│   ├── modelA/v1_main/  modelA/v2_robustness/
│   ├── modelB/  modelC/  integration/
├── figures/
│   ├── scripts/                    # code that draws Figures 1–12 of the article
│   └── output/                     # Figures 1–12 (300-dpi PNG and vector PDF)
├── docs/simulator_parameters.md    # every simulation parameter and its source
├── data/README.md                  # how to obtain each dataset (no data are redistributed)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Option 1 – Reproduce the figures from the saved results (CPU, about 15 minutes)

```bash
git clone https://github.com/<your-username>/sahay-early-warning.git
cd sahay-early-warning
pip install -r requirements.txt

cd figures/scripts
python fig_arch.py        # Figure 1
python fig_models.py      # Figure 2
python repro.py           # rebuilds the simulated cohort and refits XGBoost/LR (seed 42) -> repro.pkl
python figs_data.py       # Figures 3–12 from results/ and repro.pkl
```

Figures are written to `figures/output/`. `repro.py` runs on CPU, so XGBoost values can differ from the GPU run by less than 0.01 AUPRC (noted in the Fig. 5 caption).

## Option 2 – Re-run the experiments on Kaggle (GPU T4 × 2)

Run the notebooks in this order. For each one: **Kaggle → Create → New Notebook → File → Import Notebook**, choose the `.ipynb`, set **Accelerator = GPU T4 × 2**, then **Save Version → Save & Run All (Commit)**. Outputs appear in the notebook's **Output** tab.

| Step | Notebook | Settings and inputs | Approx. time |
|---|---|---|---|
| 1 | `01_ModelC_risk_predictor.ipynb` | Internet off; no dataset needed (the cohort is simulated) | ~1–2 h |
| 2 | `02_ModelA_text.ipynb` | **Internet on** (downloads MuRIL and the public text datasets from GitHub) | up to ~11 h (use `02b` for a shorter run) |
| 3 | `03_ModelB_voice.ipynb` | **Internet on**; add the Androids Corpus as a **private** Kaggle dataset (*Add Input*); add a Kaggle secret **`HF_TOKEN`** (*Add-ons → Secrets*) from a Hugging Face account that has accepted the EmoTa terms | ~30–60 min |
| 4 | `04_Integration_A_B_to_C.ipynb` | Internet off; uses Model A's confusion matrix and Model B's sensitivity/specificity, already entered in the notebook (`MODEL_A_CM`, `MODEL_B`) | ~1–2 h |

If you re-train Model A or B, paste your new values into `MODEL_A_CM` and `MODEL_B` in step 4.

**Seeds:** 42, 7 and 2026 for all final models; the simulated cohort uses seed 2026. **Splits:** victim-level (Model C) and speaker-level (Model B), so no person appears in both training and test data.

---

## Key results (held-out test set, mean of 3 seeds)

| Result | Value |
|---|---|
| Model C, Attention-LSTM, PSY+LEGAL | AUROC 0.935, AUPRC 0.403 (chance 0.032) |
| Gain from legal-process features (XGBoost) | AUPRC +0.179 (95% CI 0.156–0.201), p < 0.001 |
| Negative control (no legal effects) | AUPRC −0.000 (95% CI −0.008 to 0.007) |
| Early detection of escalation episodes | 76% → 84%, median lead 3 weeks |
| Unified system with real Model A/B errors | AUPRC 0.353 vs 0.141 for the rule-based score |
| Model A, sentiment / cause | macro-F1 0.549 / 0.509 |
| Model B, depression (speaker level) | macro-F1 0.853, AUROC 0.943 |

Full tables are in `results/`.

---

## Data

No dataset is included in this repository. See [`data/README.md`](data/README.md) for where to obtain each one. The Androids Corpus and EmoTa have licences that **do not allow redistribution**: keep them as private Kaggle datasets and never commit them here. Never commit a Hugging Face token; store it only as a Kaggle secret.

## Ethics

The study did not involve human participants. Any real deployment would need informed consent, ethics approval, compliance with India's Digital Personal Data Protection Act, 2023, and human review of every alert (Section 6.4 of the article).

## Use of AI tools

Claude (Anthropic) assisted with writing and debugging code, drawing figures from the results and drafting text. The authors reviewed all code and results.

## Citation

If you use this code, please cite the article (details in `CITATION.cff`, updated on publication).

## Licence

Code: MIT (see `LICENSE`). Datasets remain under their own licences.
