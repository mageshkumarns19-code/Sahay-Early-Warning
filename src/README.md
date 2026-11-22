# Python scripts

Plain-Python versions of the notebooks (cells marked with `# %%`).

| Script | Notebook |
|---|---|
| `sahay_modelC.py` | `01_ModelC_risk_predictor.ipynb` |
| `sahay_modelA_v2.py` | `02b_ModelA_text_v2_robustness.ipynb` (the main run, `02_ModelA_text.ipynb`, uses batch 16, encoder LR 2e-5 and up to 6 epochs, without the time guard and per-run saving) |
| `sahay_modelB.py` | `03_ModelB_voice.ipynb` |
| `sahay_integration_ABC.py` | `04_Integration_A_B_to_C.ipynb` |

`figures/scripts/repro.py` imports `sahay_modelC.py` to rebuild the simulated cohort.
