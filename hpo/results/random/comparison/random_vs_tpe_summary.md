## Teljességi ellenőrzés

| Architektúra | Trial-keret | Érvényes trial, RS / TPE | FAIL, RS | Paraméter | Paraméterszám egyezik | Azonos 0–9. trial: paraméter / állapot / érték | Azonos paraméterű trial a 10.-től | Türelem 30 |
|---|---:|---:|---:|---:|:---:|---:|---:|:---:|
| MLPPricer | 25 | 25 / 25 | 0 | 30 901 | igen | 10 / 10 / 10 | 0 | igen |
| DeepMLPPricer | 30 | 30 / 30 | 0 | 267 265 | igen | 10 / 10 / 10 | 0 | igen |
| DenseMLPPricer | 30 | 30 / 30 | 0 | 101 381 | igen | 10 / 10 / 10 | 0 | igen |
| ResNetPricer | 40 | 40 / 40 | 0 | 399 873 | igen | 10 / 10 / 10 | 0 | igen |
| GELUResNetPricer | 40 | 40 / 40 | 1 | 398 337 | igen | 10 / 10 / 10 | 0 | igen |
| HighwayPricer | 40 | 40 / 40 | 0 | 527 873 | igen | 10 / 10 / 10 | 0 | igen |
| FINNPricer | 40 | 40 / 40 | 0 | 402 882 | igen | 10 / 10 / 10 | 0 | igen |

## Végső tanítás (p30)

| Architektúra | Modell | Val MSE | Teszt-RMSE | Teszt-MAE | Teszt-R² | Teszt max. hiba | Legjobb / futott epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | kiindulási legjobb (v1) | 2.06e-5 | 4.60e-3 | 2.79e-3 | 0.999449 | 8.17e-2 | 123 / 133 |
| MLPPricer | RS p30 | 7.73e-7 | 9.05e-4 | 5.50e-4 | 0.999979 | 3.55e-2 | 198 / 200 |
| MLPPricer | TPE p30 | 9.01e-7 | 9.78e-4 | 6.19e-4 | 0.999975 | 3.47e-2 | 199 / 200 |
| DeepMLPPricer | kiindulási legjobb (v1) | 3.63e-5 | 6.06e-3 | 4.35e-3 | 0.999044 | 7.15e-2 | 27 / 37 |
| DeepMLPPricer | RS p30 | 8.61e-6 | 2.95e-3 | 2.19e-3 | 0.999774 | 4.03e-2 | 56 / 86 |
| DeepMLPPricer | TPE p30 | 4.56e-6 | 2.14e-3 | 1.53e-3 | 0.999881 | 4.25e-2 | 145 / 175 |
| DenseMLPPricer | kiindulási legjobb (v1) | 3.23e-4 | 1.80e-2 | 1.31e-2 | 0.991528 | 1.35e-1 | 5 / 15 |
| DenseMLPPricer | RS p30 | 1.86e-5 | 4.37e-3 | 2.85e-3 | 0.999502 | 6.93e-2 | 55 / 85 |
| DenseMLPPricer | TPE p30 | 4.03e-6 | 2.03e-3 | 1.32e-3 | 0.999892 | 4.24e-2 | 113 / 143 |
| ResNetPricer | kiindulási legjobb (v2) | 8.35e-6 | 2.91e-3 | 2.18e-3 | 0.999779 | 5.27e-2 | 182 / 200 |
| ResNetPricer | RS p30 | 1.55e-5 | 3.98e-3 | 3.05e-3 | 0.999587 | 5.70e-2 | 178 / 200 |
| ResNetPricer | TPE p30 | 4.00e-6 | 2.04e-3 | 1.35e-3 | 0.999892 | 5.22e-2 | 132 / 162 |
| GELUResNetPricer | kiindulási legjobb (v1) | 8.50e-5 | 9.29e-3 | 6.25e-3 | 0.997752 | 9.72e-2 | 31 / 41 |
| GELUResNetPricer | RS p30 | 1.54e-5 | 3.97e-3 | 2.71e-3 | 0.999589 | 5.50e-2 | 21 / 51 |
| GELUResNetPricer | TPE p30 | 1.48e-5 | 3.88e-3 | 2.63e-3 | 0.999609 | 5.51e-2 | 21 / 51 |
| HighwayPricer | kiindulási legjobb (v1) | 2.20e-4 | 1.49e-2 | 1.06e-2 | 0.994234 | 1.23e-1 | 9 / 19 |
| HighwayPricer | RS p30 | 9.21e-6 | 3.06e-3 | 2.01e-3 | 0.999756 | 4.71e-2 | 194 / 200 |
| HighwayPricer | TPE p30 | 8.53e-6 | 2.94e-3 | 1.99e-3 | 0.999774 | 4.46e-2 | 179 / 200 |
| FINNPricer | kiindulási legjobb (v1) | 8.24e-5 | 9.14e-3 | 6.13e-3 | 0.997824 | 9.60e-2 | 21 / 31 |
| FINNPricer | RS p30 | 1.45e-5 | 3.81e-3 | 2.95e-3 | 0.999622 | 5.15e-2 | 33 / 63 |
| FINNPricer | TPE p30 | 8.67e-6 | 2.98e-3 | 1.96e-3 | 0.999769 | 4.05e-2 | 77 / 107 |

## Relatív különbségek (p30)

| Architektúra | Val MSE, RS a TPE-hez | Teszt-RMSE, RS a TPE-hez | Val MSE-arány RS / TPE | Teszt-RMSE-arány RS / TPE | Val MSE, RS a kiindulási legjobbhoz | Val MSE, TPE a kiindulási legjobbhoz | Jobb (val MSE) |
|---|---:|---:|---:|---:|---:|---:|:---:|
| MLPPricer | −14.2% | −7.5% | 0.86 | 0.93 | −96.2% | −95.6% | RS |
| DeepMLPPricer | +89.1% | +37.7% | 1.89 | 1.38 | −76.3% | −87.5% | TPE |
| DenseMLPPricer | +361.3% | +114.9% | 4.61 | 2.15 | −94.2% | −98.8% | TPE |
| ResNetPricer | +287.8% | +95.5% | 3.88 | 1.96 | +85.6% | −52.1% | TPE |
| GELUResNetPricer | +4.5% | +2.4% | 1.05 | 1.02 | −81.8% | −82.6% | TPE |
| HighwayPricer | +8.0% | +4.0% | 1.08 | 1.04 | −95.8% | −96.1% | TPE |
| FINNPricer | +67.0% | +28.0% | 1.67 | 1.28 | −82.4% | −89.5% | TPE |
| Medián | – | – | 1.67 | 1.28 | – | – | – |

## Rangsor (p30 val MSE, 1 = legjobb)

| Architektúra | RS p30 | TPE p30 |
|---|---:|---:|
| MLPPricer | 1 | 1 |
| DeepMLPPricer | 2 | 4 |
| DenseMLPPricer | 7 | 3 |
| ResNetPricer | 6 | 2 |
| GELUResNetPricer | 5 | 7 |
| HighwayPricer | 3 | 5 |
| FINNPricer | 4 | 6 |

## Kendall τ a két rangsor között

| Rangsor | τ | p |
|---|---:|---:|
| p30 val MSE, RS vs. TPE | 0.238 | 0.562 |

## Páros összevetés (7 architektúra)

| Mutató | A TPE jobb | Előjelpróba, p | Wilcoxon, W | Wilcoxon, p | RS / TPE medián | RS / TPE mértani átlag |
|---|---:|---:|---:|---:|---:|---:|
| Legjobb keresési val MSE | 6/7 | 0.125 | 1 | 0.031 | 1.29 | 1.61 |
| Végső p30 val MSE | 6/7 | 0.125 | 3 | 0.078 | 1.67 | 1.77 |

## Keresés: legjobb keresési val MSE

| Architektúra | Közös indítás legjobbja | Legjobb, RS (trial) | Legjobb, TPE (trial) | RS / TPE | RS eléri a TPE legjobbját | TPE eléri az RS legjobbját | Teljes keresés, RS / TPE (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 2.72e-6 | 1.81e-6 (#14) | 1.46e-6 (#15) | 1.24 | nem éri el | #15 (16. érvényes trial, 180 s, 346 epoch) | 167 / 266 |
| DeepMLPPricer | 2.78e-5 | 2.78e-5 (#9) | 8.40e-6 (#19) | 3.30 | nem éri el | #9 (10. érvényes trial, 139 s, 134 epoch) | 137 / 412 |
| DenseMLPPricer | 1.51e-5 | 1.51e-5 (#1) | 5.79e-6 (#25) | 2.60 | nem éri el | #1 (2. érvényes trial, 40 s, 42 epoch) | 211 / 429 |
| ResNetPricer | 6.83e-5 | 1.35e-5 (#27) | 1.04e-5 (#35) | 1.29 | nem éri el | #35 (36. érvényes trial, 941 s, 623 epoch) | 493 / 1004 |
| GELUResNetPricer | 2.34e-5 | 8.73e-6 (#36) | 4.65e-6 (#13) | 1.88 | nem éri el | #13 (14. érvényes trial, 177 s, 192 epoch) | 358 / 598 |
| HighwayPricer | 1.01e-5 | 1.01e-5 (#3) | 9.00e-6 (#38) | 1.12 | nem éri el | #3 (4. érvényes trial, 135 s, 111 epoch) | 389 / 801 |
| FINNPricer | 2.62e-5 | 1.64e-5 (#33) | 1.70e-5 (#19) | 0.96 | #33 (34. érvényes trial, 316 s, 393 epoch) | nem éri el | 369 / 536 |

## Pruning

| Architektúra | Módszer | COMPLETE / PRUNED | COMPLETE-arány | Átl. epoch / trial | Összes epoch | 10. trialtól: COMPLETE / összes | 10. trialtól: átl. epoch | 10. trialtól: 40 epochig futott | Jobb a közös indításnál | ≤ 2× a legjobb |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | RS | 10 / 15 | 40% | 19.1 | 477 | 4 / 15 | 17.7 | 3 | 3 | 3 |
| MLPPricer | TPE | 10 / 15 | 40% | 20.0 | 500 | 4 / 15 | 19.3 | 4 | 3 | 3 |
| DeepMLPPricer | RS | 5 / 25 | 17% | 8.0 | 241 | 1 / 20 | 5.3 | 0 | 0 | 0 |
| DeepMLPPricer | TPE | 7 / 23 | 23% | 13.2 | 397 | 3 / 20 | 13.2 | 1 | 2 | 1 |
| DenseMLPPricer | RS | 11 / 19 | 37% | 14.3 | 430 | 5 / 20 | 13.1 | 1 | 0 | 0 |
| DenseMLPPricer | TPE | 11 / 19 | 37% | 16.6 | 498 | 5 / 20 | 16.5 | 4 | 4 | 3 |
| ResNetPricer | RS | 10 / 30 | 25% | 14.1 | 564 | 6 / 30 | 13.5 | 3 | 5 | 2 |
| ResNetPricer | TPE | 13 / 27 | 32% | 16.8 | 671 | 9 / 30 | 17.0 | 1 | 9 | 1 |
| GELUResNetPricer | RS | 12 / 28 | 30% | 11.7 | 468 | 7 / 30 | 11.0 | 1 | 2 | 1 |
| GELUResNetPricer | TPE | 11 / 29 | 28% | 16.6 | 665 | 6 / 30 | 17.6 | 2 | 4 | 3 |
| HighwayPricer | RS | 11 / 29 | 28% | 10.9 | 437 | 6 / 30 | 9.5 | 1 | 0 | 1 |
| HighwayPricer | TPE | 12 / 28 | 30% | 17.1 | 682 | 7 / 30 | 17.6 | 7 | 1 | 7 |
| FINNPricer | RS | 9 / 31 | 22% | 11.5 | 460 | 5 / 30 | 10.5 | 0 | 1 | 1 |
| FINNPricer | TPE | 6 / 34 | 15% | 12.5 | 500 | 2 / 30 | 11.9 | 2 | 2 | 2 |
| Összesen | RS | 68 / 177 | 28% | 12.6 | 3077 | 34 / 175 | 11.3 | – | 11 | 8 |
| Összesen | TPE | 70 / 175 | 29% | 16.0 | 3913 | 36 / 175 | 16.0 | – | 25 | 20 |

## Legjobb hiperparaméterek

| Architektúra | Módszer | lr | weight_decay | Dekád az alsó határtól | dropout | Top-5 weight_decay (medián) | weight_decay < 1e-5 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | RS | 1.53e-3 | 1.53e-6 | 0.19 | – | 3.61e-6 | 32% |
| MLPPricer | TPE | 1.39e-3 | 1.36e-6 | 0.13 | – | 1.61e-6 | 56% |
| DeepMLPPricer | RS | 1.07e-3 | 2.34e-4 | 2.37 | 0.023 | 7.07e-6 | 30% |
| DeepMLPPricer | TPE | 1.50e-3 | 1.38e-6 | 0.14 | 0.018 | 5.40e-6 | 40% |
| DenseMLPPricer | RS | 1.58e-3 | 4.21e-6 | 0.62 | 0.078 | 4.21e-6 | 30% |
| DenseMLPPricer | TPE | 3.05e-3 | 1.08e-6 | 0.03 | 0.012 | 1.83e-6 | 57% |
| ResNetPricer | RS | 1.76e-3 | 2.11e-5 | 1.32 | 0.032 | 7.13e-4 | 28% |
| ResNetPricer | TPE | 4.50e-3 | 2.90e-4 | 2.46 | 0.325 | 5.24e-4 | 8% |
| GELUResNetPricer | RS | 2.87e-4 | 2.03e-6 | 0.31 | 0.145 | 4.21e-6 | 25% |
| GELUResNetPricer | TPE | 2.39e-4 | 1.10e-6 | 0.04 | 0.122 | 1.32e-6 | 65% |
| HighwayPricer | RS | 2.61e-3 | 1.21e-6 | 0.08 | 0.485 | 2.46e-6 | 28% |
| HighwayPricer | TPE | 2.21e-3 | 1.02e-6 | 0.01 | 0.406 | 1.09e-6 | 62% |
| FINNPricer | RS | 1.64e-4 | 1.34e-6 | 0.13 | 0.318 | 4.21e-6 | 28% |
| FINNPricer | TPE | 3.37e-3 | 1.07e-6 | 0.03 | 0.027 | 1.73e-6 | 60% |

## Hiperparaméter-fontosság (PED-ANOVA)

| Architektúra | RS: lr / wd / dropout | RS sorrend | TPE: lr / wd / dropout | TPE sorrend | Azonos sorrend |
|---|---|---|---|---|:---:|
| MLPPricer | 0.54 / 0.46 / – | lr > wd | 0.58 / 0.42 / – | lr > wd | igen |
| DeepMLPPricer | 0.25 / 0.45 / 0.31 | wd > dropout > lr | 0.38 / 0.43 / 0.19 | wd > lr > dropout | nem |
| DenseMLPPricer | 0.22 / 0.76 / 0.02 | wd > lr > dropout | 0.07 / 0.41 / 0.52 | dropout > wd > lr | nem |
| ResNetPricer | 0.31 / 0.33 / 0.36 | dropout > wd > lr | 0.84 / 0.13 / 0.03 | lr > wd > dropout | nem |
| GELUResNetPricer | 0.02 / 0.79 / 0.20 | wd > dropout > lr | 0.02 / 0.94 / 0.05 | wd > dropout > lr | igen |
| HighwayPricer | 0.00 / 0.98 / 0.02 | wd > dropout > lr | 0.29 / 0.66 / 0.05 | wd > lr > dropout | nem |
| FINNPricer | 0.28 / 0.43 / 0.29 | wd > dropout > lr | 0.33 / 0.39 / 0.29 | wd > lr > dropout | nem |

## Futásidő (s)

| Architektúra | Keresés, RS | Trialidő, RS | Első–utolsó trial, RS | Trialidő, TPE | Első–utolsó trial, TPE | Trialidő-arány RS / TPE | Végső tanítás, RS | Végső tanítás, TPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 167 | 167 | 167 | 266 | 266 | 0.63 | 1168 | 1161 |
| DeepMLPPricer | 137 | 137 | 137 | 412 | 413 | 0.33 | 533 | 1015 |
| DenseMLPPricer | 211 | 211 | 211 | 429 | 429 | 0.49 | 511 | 811 |
| ResNetPricer | 494 | 493 | 494 | 1004 | 1004 | 0.49 | 2032 | 964 |
| GELUResNetPricer | 359 | 358 | 1586 | 598 | 598 | 0.60 | 324 | 305 |
| HighwayPricer | 389 | 389 | 389 | 801 | 801 | 0.49 | 1300 | 1221 |
| FINNPricer | 369 | 369 | 369 | 536 | 536 | 0.69 | 401 | 637 |
| Összesen | 2127 | 2123 | – | 4045 | 4048 | 0.52 | 6268 | 6113 |

## Gépsebesség: a közös 0–9. trial ideje

| Architektúra | Epoch, 0–9. trial | Idő, 0–9. trial, TPE / RS (s) | Időarány RS / TPE | s / epoch, TPE | s / epoch, RS | Keresési epoch, TPE / RS | Epoch-arány RS / TPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 211 | 112 / 75 | 0.67 | 0.53 | 0.35 | 500 / 477 | 0.95 |
| DeepMLPPricer | 134 | 139 / 77 | 0.55 | 1.04 | 0.57 | 397 / 241 | 0.61 |
| DenseMLPPricer | 168 | 146 / 81 | 0.55 | 0.86 | 0.49 | 498 / 430 | 0.86 |
| ResNetPricer | 160 | 240 / 138 | 0.57 | 1.50 | 0.87 | 671 / 564 | 0.84 |
| GELUResNetPricer | 137 | 127 / 121 | 0.95 | 0.90 | 0.77 | 665 / 468 | 0.70 |
| HighwayPricer | 153 | 188 / 134 | 0.71 | 1.17 | 0.89 | 682 / 437 | 0.64 |
| FINNPricer | 144 | 157 / 117 | 0.75 | 1.07 | 0.80 | 500 / 460 | 0.92 |
| Összesen | 1107 | 1109 / 742 | 0.67 | 1.03 | 0.69 | 3913 / 3077 | 0.79 |

## Ellenőrzések

| Ellenőrzés | Eredmény |
|---|---|
| `*_ERROR.log` a `hpo/logs/random/` alatt | nincs |
| A legjobb keresési érték = a `trials_{arch}.csv` minimum `value`-ja (7 architektúra, TPE és RS) | igen |
| A legjobb keresési érték = a `best_params_{arch}.json` `search_best_val_loss` mezője (TPE és RS) | igen (legnagyobb relatív eltérés 0.0e+00) |
| A kiindulási legjobb val MSE = a `training_history_*.json` `best_val_loss` mezője | igen |
| Az RS `best_params_{arch}.json` `tpe_p30_*` mezői = a `hpo/results/p30/` értékei | igen |
