## Keresés

| Architektúra | Trial | COMPLETE / PRUNED, RS | COMPLETE / PRUNED, TPE | Legjobb keresési val MSE, RS (trial) | Legjobb keresési val MSE, TPE (trial) | Azonos 0–9. trial |
|---|---:|---:|---:|---:|---:|:---:|
| MLPPricer | 25 | 10 / 15 | 10 / 15 | 1.81e-6 (#14) | 1.46e-6 (#15) | igen |
| DeepMLPPricer | 30 | 5 / 25 | 7 / 23 | 2.78e-5 (#9) | 8.40e-6 (#19) | igen |
| DenseMLPPricer | 30 | 11 / 19 | 11 / 19 | 1.51e-5 (#1) | 5.79e-6 (#25) | igen |
| ResNetPricer | 40 | 10 / 30 | 13 / 27 | 1.35e-5 (#27) | 1.04e-5 (#35) | igen |
| GELUResNetPricer | 40 (+1 FAIL) | 12 / 28 | 11 / 29 | 8.73e-6 (#36) | 4.65e-6 (#13) | igen |
| HighwayPricer | 40 | 11 / 29 | 12 / 28 | 1.01e-5 (#3) | 9.00e-6 (#38) | igen |
| FINNPricer | 40 | 9 / 31 | 6 / 34 | 1.64e-5 (#33) | 1.70e-5 (#19) | igen |

## Legjobb hiperparaméterek

| Architektúra | lr, RS | weight_decay, RS | dropout, RS | lr, TPE | weight_decay, TPE | dropout, TPE |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 1.53e-3 | 1.53e-6 | – | 1.39e-3 | 1.36e-6 | – |
| DeepMLPPricer | 1.07e-3 | 2.34e-4 | 0.023 | 1.50e-3 | 1.38e-6 | 0.018 |
| DenseMLPPricer | 1.58e-3 | 4.21e-6 | 0.078 | 3.05e-3 | 1.08e-6 | 0.012 |
| ResNetPricer | 1.76e-3 | 2.11e-5 | 0.032 | 4.50e-3 | 2.90e-4 | 0.325 |
| GELUResNetPricer | 2.87e-4 | 2.03e-6 | 0.145 | 2.39e-4 | 1.10e-6 | 0.122 |
| HighwayPricer | 2.61e-3 | 1.21e-6 | 0.485 | 2.21e-3 | 1.02e-6 | 0.406 |
| FINNPricer | 1.64e-4 | 1.34e-6 | 0.318 | 3.37e-3 | 1.07e-6 | 0.027 |

## Végső tanítás (p30)

| Architektúra | Paraméter | Teszt-RMSE, RS | Teszt-RMSE, TPE | Val MSE, RS | Val MSE, TPE | Val MSE változása, RS a TPE-hez | Legjobb / futott epoch, RS |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 9.05e-4 | 9.78e-4 | 7.73e-7 | 9.01e-7 | −14.2% | 198 / 200 |
| DeepMLPPricer | 267 265 | 2.95e-3 | 2.14e-3 | 8.61e-6 | 4.56e-6 | +89.1% | 56 / 86 |
| DenseMLPPricer | 101 381 | 4.37e-3 | 2.03e-3 | 1.86e-5 | 4.03e-6 | +361.3% | 55 / 85 |
| ResNetPricer | 399 873 | 3.98e-3 | 2.04e-3 | 1.55e-5 | 4.00e-6 | +287.8% | 178 / 200 |
| GELUResNetPricer | 398 337 | 3.97e-3 | 3.88e-3 | 1.54e-5 | 1.48e-5 | +4.5% | 21 / 51 |
| HighwayPricer | 527 873 | 3.06e-3 | 2.94e-3 | 9.21e-6 | 8.53e-6 | +8.0% | 194 / 200 |
| FINNPricer | 402 882 | 3.81e-3 | 2.98e-3 | 1.45e-5 | 8.67e-6 | +67.0% | 33 / 63 |

## Futásidő (s)

| Architektúra | Keresés, RS | Trialidők összege, RS | Trialidők összege, TPE | Végső tanítás, RS | Végső tanítás, TPE | Összesen, RS |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 167 | 167 | 266 | 1168 | 1161 | 1335 |
| DeepMLPPricer | 137 | 137 | 412 | 533 | 1015 | 670 |
| DenseMLPPricer | 211 | 211 | 429 | 511 | 811 | 722 |
| ResNetPricer | 494 | 493 | 1004 | 2032 | 964 | 2525 |
| GELUResNetPricer | 174 | 358 | 598 | 324 | 305 | 498 |
| HighwayPricer | 389 | 389 | 801 | 1300 | 1221 | 1689 |
| FINNPricer | 369 | 369 | 536 | 401 | 637 | 770 |
| Összesen | 1941 | 2123 | 4045 | 6268 | 6113 | 8210 |
