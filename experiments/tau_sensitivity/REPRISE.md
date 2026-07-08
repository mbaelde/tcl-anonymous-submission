# Reprise tau_sensitivity

Commande de reprise (depuis `D:\repos\tcl-code`) :

```powershell
Set-Location D:\repos\tcl-code; $env:PYTHONUTF8 = '1'; py -3.14 -m uv run python -m experiments.tau_sensitivity.run --config experiments/tau_sensitivity/config.yaml --parallel 1 --skip-existing
```

**État au 2026-05-25** : 18 cellules TCL complètes, 18 cellules lagrangian en cours (interrompues). Le `--skip-existing` reprend uniquement les cellules sans `result.txt`.

**Suite** : une fois terminé, ajouter §7.1.11 (tau sensitivity) dans `TCL_paper_draft.md` entre §7.1.10 et §7.1.12, puis `py -3.14 migrate_to_tmlr.py` + `latexmk -pdf -g TCL_paper_tmlr.tex`.
