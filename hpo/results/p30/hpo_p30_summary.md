## Teszt-RMSE és validációs MSE

| Architektúra | Paraméter | Teszt-RMSE, kiindulási | Teszt-RMSE, HPO p30 | Változás | Val MSE, kiindulási | Val MSE, HPO p30 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 4.60e-3 | 9.78e-4 | −78.7% | 2.06e-5 | 9.01e-7 |
| DeepMLPPricer | 267 265 | 6.06e-3 | 2.14e-3 | −64.7% | 3.63e-5 | 4.56e-6 |
| DenseMLPPricer | 101 381 | 1.80e-2 | 2.03e-3 | −88.7% | 3.23e-4 | 4.03e-6 |
| ResNetPricer | 399 873 | 6.22e-3 | 2.04e-3 | −67.2% | 3.85e-5 | 4.00e-6 |
| GELUResNetPricer | 398 337 | 9.29e-3 | 3.88e-3 | −58.3% | 8.50e-5 | 1.48e-5 |
| HighwayPricer | 527 873 | 1.49e-2 | 2.94e-3 | −80.2% | 2.20e-4 | 8.53e-6 |
| FINNPricer | 402 882 | 9.14e-3 | 2.98e-3 | −67.4% | 8.24e-5 | 8.67e-6 |

## Hiperparaméterek

| Architektúra | Trial | lr | weight_decay | dropout |
|---|---:|---:|---:|---:|
| MLPPricer | 25 | 1.39e-3 | 1.36e-6 | – |
| DeepMLPPricer | 30 | 1.50e-3 | 1.38e-6 | 0.018 |
| DenseMLPPricer | 30 | 3.05e-3 | 1.08e-6 | 0.012 |
| ResNetPricer | 40 | 4.50e-3 | 2.90e-4 | 0.325 |
| GELUResNetPricer | 40 | 2.39e-4 | 1.10e-6 | 0.122 |
| HighwayPricer | 40 | 2.21e-3 | 1.02e-6 | 0.406 |
| FINNPricer | 40 | 3.37e-3 | 1.07e-6 | 0.027 |

## Kiindulási legjobb, HPO p10 és HPO p30

| Architektúra | Modell | Val MSE | Teszt-RMSE | Teszt max. hiba | Legjobb / futott epoch |
|---|---|---:|---:|---:|---:|
| MLPPricer | kiindulási legjobb (v1) | 2.06e-5 | 4.60e-3 | 8.17e-2 | 123 / 133 |
| MLPPricer | HPO p10 | 9.01e-7 | 9.78e-4 | 3.47e-2 | 199 / 200 |
| MLPPricer | HPO p30 | 9.01e-7 | 9.78e-4 | 3.47e-2 | 199 / 200 |
| DeepMLPPricer | kiindulási legjobb (v1) | 3.63e-5 | 6.06e-3 | 7.15e-2 | 27 / 37 |
| DeepMLPPricer | HPO p10 | 6.05e-6 | 2.47e-3 | 4.98e-2 | 58 / 68 |
| DeepMLPPricer | HPO p30 | 4.56e-6 | 2.14e-3 | 4.25e-2 | 145 / 175 |
| DenseMLPPricer | kiindulási legjobb (v1) | 3.23e-4 | 1.80e-2 | 1.35e-1 | 5 / 15 |
| DenseMLPPricer | HPO p10 | 4.06e-6 | 2.04e-3 | 4.22e-2 | 76 / 86 |
| DenseMLPPricer | HPO p30 | 4.03e-6 | 2.03e-3 | 4.24e-2 | 113 / 143 |
| ResNetPricer | kiindulási legjobb (v2) | 8.35e-6 | 2.91e-3 | 5.27e-2 | 182 / 200 |
| ResNetPricer | HPO p10 | 1.81e-5 | 4.28e-3 | 6.82e-2 | 29 / 39 |
| ResNetPricer | HPO p30 | 4.00e-6 | 2.04e-3 | 5.22e-2 | 132 / 162 |
| GELUResNetPricer | kiindulási legjobb (v1) | 8.50e-5 | 9.29e-3 | 9.72e-2 | 31 / 41 |
| GELUResNetPricer | HPO p10 | 1.48e-5 | 3.88e-3 | 5.51e-2 | 21 / 31 |
| GELUResNetPricer | HPO p30 | 1.48e-5 | 3.88e-3 | 5.51e-2 | 21 / 51 |
| HighwayPricer | kiindulási legjobb (v1) | 2.20e-4 | 1.49e-2 | 1.23e-1 | 9 / 19 |
| HighwayPricer | HPO p10 | 8.55e-6 | 2.95e-3 | 4.48e-2 | 124 / 134 |
| HighwayPricer | HPO p30 | 8.53e-6 | 2.94e-3 | 4.46e-2 | 179 / 200 |
| FINNPricer | kiindulási legjobb (v1) | 8.24e-5 | 9.14e-3 | 9.60e-2 | 21 / 31 |
| FINNPricer | HPO p10 | 8.67e-6 | 2.98e-3 | 4.05e-2 | 77 / 87 |
| FINNPricer | HPO p30 | 8.67e-6 | 2.98e-3 | 4.05e-2 | 77 / 107 |

## A HPO p30 változása (val MSE) és a tanítási idő

| Architektúra | A HPO p10-hez képest | A kiindulási legjobbhoz képest | Tanítási idő p30 (s) | Tanítási idő p10 (s) |
|---|---:|---:|---:|---:|
| MLPPricer | +0.0% | −95.6% | 1161 | 1693 |
| DeepMLPPricer | −24.7% | −87.5% | 1015 | 682 |
| DenseMLPPricer | −0.7% | −98.8% | 811 | 840 |
| ResNetPricer | −77.9% | −52.1% | 964 | 366 |
| GELUResNetPricer | +0.0% | −82.6% | 305 | 273 |
| HighwayPricer | −0.3% | −96.1% | 1221 | 1240 |
| FINNPricer | +0.0% | −89.5% | 637 | 793 |

## A p30 és a p10 tanítási görbéje a közös epochokon

| Architektúra | Közös epoch | Bitre azonos | Első eltérő epoch | Max. eltérés, train_loss | Max. eltérés, val_loss | Max. eltérés, lr |
|---|---:|:---:|---:|---:|---:|---:|
| MLPPricer | 200 | igen | – | 0 | 0 | 0 |
| DeepMLPPricer | 68 | igen | – | 0 | 0 | 0 |
| DenseMLPPricer | 86 | igen | – | 0 | 0 | 0 |
| ResNetPricer | 39 | igen | – | 0 | 0 | 0 |
| GELUResNetPricer | 31 | igen | – | 0 | 0 | 0 |
| HighwayPricer | 134 | igen | – | 0 | 0 | 0 |
| FINNPricer | 87 | igen | – | 0 | 0 | 0 |

## Checkpoint-ellenőrzés: újraszámolt és tárolt val MSE

| Architektúra | Tárolt val MSE | Újraszámolt val MSE | Relatív eltérés |
|---|---:|---:|---:|
| MLPPricer | 9.01e-7 | 9.01e-7 | 6.84e-9 |
| DeepMLPPricer | 4.56e-6 | 4.56e-6 | 1.21e-8 |
| DenseMLPPricer | 4.03e-6 | 4.03e-6 | 8.88e-8 |
| ResNetPricer | 4.00e-6 | 4.00e-6 | 2.00e-9 |
| GELUResNetPricer | 1.48e-5 | 1.48e-5 | 1.38e-8 |
| HighwayPricer | 8.53e-6 | 8.53e-6 | 1.60e-7 |
| FINNPricer | 8.67e-6 | 8.67e-6 | 6.51e-8 |
