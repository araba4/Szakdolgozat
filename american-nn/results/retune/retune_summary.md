## Teszt-RMSE három szinten

| Architektúra | Amerikai, uniform p30 | Amerikai, transfer p30 | Amerikai, retune p30 | Európai, uniform p30 | Európai, transfer p30 | Európai, retune p30 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 4.60e-3 | 7.83e-4 | 7.47e-4 | 4.45e-3 | 7.91e-4 | 7.87e-4 |
| DeepMLPPricer | 7.15e-3 | 2.23e-3 | 2.50e-3 | 1.20e-2 | 2.37e-3 | 2.30e-3 |
| DenseMLPPricer | 1.58e-2 | 2.37e-3 | 3.86e-3 | 1.56e-2 | 2.23e-3 | 3.86e-3 |
| ResNetPricer | 2.51e-3 | 2.13e-3 | 2.29e-3 | 2.33e-3 | 2.13e-3 | 2.60e-3 |
| GELUResNetPricer | 9.49e-3 | 2.28e-3 | 1.85e-3 | 9.19e-3 | 1.87e-3 | 1.73e-3 |
| HighwayPricer | 1.27e-2 | 2.92e-3 | 2.62e-3 | 1.38e-2 | 2.92e-3 | 2.50e-3 |
| FINNPricer | 9.78e-3 | 1.96e-3 | 1.18e-3 | 9.19e-3 | 1.89e-3 | 1.10e-3 |

## A retune p30 tanítása

| Architektúra | Cél | Legjobb epoch | Futott epoch | 200 epochos korlát |
|---|---|---:|---:|---:|
| MLPPricer | amerikai | 199 | 200 | igen |
| MLPPricer | európai | 196 | 200 | igen |
| DeepMLPPricer | amerikai | 126 | 156 | nem |
| DeepMLPPricer | európai | 56 | 86 | nem |
| DenseMLPPricer | amerikai | 80 | 110 | nem |
| DenseMLPPricer | európai | 76 | 106 | nem |
| ResNetPricer | amerikai | 136 | 166 | nem |
| ResNetPricer | európai | 120 | 150 | nem |
| GELUResNetPricer | amerikai | 58 | 88 | nem |
| GELUResNetPricer | európai | 56 | 86 | nem |
| HighwayPricer | amerikai | 33 | 63 | nem |
| HighwayPricer | európai | 49 | 79 | nem |
| FINNPricer | amerikai | 193 | 200 | igen |
| FINNPricer | európai | 182 | 200 | igen |

## A legjobb amerikai modellek

| Kiválasztás | Modell | Teszt-RMSE | Val MSE | Teszt R² | Teszt-RMSE ($) |
|---|---|---:|---:|---:|---:|
| A három szint közül | MLPPricer, retune p30 | 7.47e-4 | 5.25e-7 | 0.999977 | 0.0755 |
| A retune p30 szinten | MLPPricer, retune p30 | 7.47e-4 | 5.25e-7 | 0.999977 | 0.0755 |

## A retune p30 változása a transfer p30-hoz és a uniform p30-hoz képest

| Architektúra | Cél | Teszt-RMSE, a transferhez | Transfer és retune aránya | Teszt-RMSE, a uniformhoz | Uniform és retune aránya | Val MSE, a transferhez | Val MSE, a uniformhoz |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | amerikai | −4.5% | 1.05 | −83.7% | 6.15 | −9.9% | −97.4% |
| MLPPricer | európai | −0.6% | 1.01 | −82.3% | 5.66 | −1.6% | −97.0% |
| DeepMLPPricer | amerikai | +11.7% | 0.89 | −65.1% | 2.86 | +26.2% | −87.8% |
| DeepMLPPricer | európai | −3.0% | 1.03 | −80.9% | 5.23 | −5.8% | −96.4% |
| DenseMLPPricer | amerikai | +63.0% | 0.61 | −75.6% | 4.09 | +165.0% | −94.1% |
| DenseMLPPricer | európai | +73.4% | 0.58 | −75.3% | 4.04 | +200.6% | −93.9% |
| ResNetPricer | amerikai | +7.4% | 0.93 | −8.9% | 1.10 | +15.6% | −17.1% |
| ResNetPricer | európai | +22.1% | 0.82 | +11.3% | 0.90 | +49.4% | +22.7% |
| GELUResNetPricer | amerikai | −18.7% | 1.23 | −80.5% | 5.12 | −33.7% | −96.2% |
| GELUResNetPricer | európai | −7.5% | 1.08 | −81.1% | 5.30 | −14.4% | −96.5% |
| HighwayPricer | amerikai | −10.3% | 1.12 | −79.5% | 4.87 | −20.0% | −95.8% |
| HighwayPricer | európai | −14.5% | 1.17 | −81.9% | 5.54 | −26.9% | −96.8% |
| FINNPricer | amerikai | −39.8% | 1.66 | −87.9% | 8.29 | −63.6% | −98.6% |
| FINNPricer | európai | −42.1% | 1.73 | −88.1% | 8.39 | −66.4% | −98.6% |
| Medián, amerikai |  | −4.5% | 1.05 | −79.5% |  |  |  |
| Medián, európai |  | −3.0% | 1.03 | −81.1% |  |  |  |

## Ahol a retune p30 rosszabb a transfer p30-nál

| Cél | Architektúra |
|---|---|
| amerikai | DeepMLPPricer, DenseMLPPricer, ResNetPricer |
| európai | DenseMLPPricer, ResNetPricer |

## A transfer és a retune konfiguráció a romlott futásoknál

| Architektúra | Cél | Transfer: lr / wd / dropout | Retune: lr / wd / dropout | Amerikai keresés legjobbja (trial) | Amerikai keresés 2. (trial) | Európai call keresés legjobbja (trial) | p30 val MSE, transfer | p30 val MSE, retune |
|---|---|---|---|---:|---:|---:|---:|---:|
| DeepMLPPricer | amerikai | 1.50e-3 / 1.38e-6 / 0.018 | 1.72e-4 / 1.64e-6 / 0.016 | 7.22e-6 (#16) | 2.70e-5 (#21) | 8.40e-6 (#19) | 4.95e-6 | 6.25e-6 |
| DenseMLPPricer | amerikai | 3.05e-3 / 1.08e-6 / 0.012 | 1.90e-3 / 2.26e-6 / 0.162 | 1.62e-5 (#16) | 1.78e-5 (#1) | 5.79e-6 (#25) | 5.53e-6 | 1.46e-5 |
| DenseMLPPricer | európai | 3.05e-3 / 1.08e-6 / 0.012 | 1.90e-3 / 2.26e-6 / 0.162 | 1.62e-5 (#16) | 1.78e-5 (#1) | 5.79e-6 (#25) | 4.91e-6 | 1.48e-5 |
| ResNetPricer | amerikai | 4.50e-3 / 2.90e-4 / 0.325 | 8.60e-3 / 6.30e-4 / 0.104 | 1.48e-5 (#10) | 1.52e-5 (#9) | 1.04e-5 (#35) | 4.41e-6 | 5.10e-6 |
| ResNetPricer | európai | 4.50e-3 / 2.90e-4 / 0.325 | 8.60e-3 / 6.30e-4 / 0.104 | 1.48e-5 (#10) | 1.52e-5 (#9) | 1.04e-5 (#35) | 4.36e-6 | 6.52e-6 |

## Hiperparaméterek: európai call TPE p30 és amerikai put retune

| Architektúra | Európai lr | Amerikai lr | Amerikai és európai lr aránya | Európai wd | Európai wd, dekád 1e-6 felett | Amerikai wd | Amerikai wd, dekád 1e-6 felett | Amerikai wd < 2.51e-6 | Európai dropout | Amerikai dropout | Top-5 wd log-mediánja, európai | Top-5 wd log-mediánja, amerikai |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 1.39e-3 | 2.73e-3 | 1.97 | 1.36e-6 | 0.13 | 1.23e-6 | 0.09 | igen | – | – | 1.61e-6 | 1.46e-6 |
| DeepMLPPricer | 1.50e-3 | 1.72e-4 | 0.11 | 1.38e-6 | 0.14 | 1.64e-6 | 0.21 | igen | 0.018 | 0.016 | 5.40e-6 | 3.56e-6 |
| DenseMLPPricer | 3.05e-3 | 1.90e-3 | 0.62 | 1.08e-6 | 0.03 | 2.26e-6 | 0.35 | igen | 0.012 | 0.162 | 1.83e-6 | 3.12e-6 |
| ResNetPricer | 4.50e-3 | 8.60e-3 | 1.91 | 2.90e-4 | 2.46 | 6.30e-4 | 2.80 | nem | 0.325 | 0.104 | 5.24e-4 | 5.10e-4 |
| GELUResNetPricer | 2.39e-4 | 1.66e-4 | 0.69 | 1.10e-6 | 0.04 | 1.10e-6 | 0.04 | igen | 0.122 | 0.069 | 1.32e-6 | 1.50e-6 |
| HighwayPricer | 2.21e-3 | 2.39e-3 | 1.08 | 1.02e-6 | 0.01 | 1.08e-6 | 0.03 | igen | 0.406 | 0.128 | 1.09e-6 | 1.14e-6 |
| FINNPricer | 3.37e-3 | 2.36e-4 | 0.07 | 1.07e-6 | 0.03 | 1.22e-6 | 0.09 | igen | 0.027 | 0.039 | 1.73e-6 | 1.32e-6 |

## A hiperparaméterek összesítése

| Mutató | Érték |
|---|---:|
| Amerikai retune: wd < 2.51e-6 | 6/7 |
| Európai call TPE: wd < 2.51e-6 | 6/7 |
| Az amerikai és az európai érték log10-különbségének abszolút mediánja, lr | 0.28 |
| Az amerikai és az európai érték log10-különbségének abszolút mediánja, wd | 0.06 |

## A keresés lefolyása

| Architektúra | Kampány | Érvényes trial / cél | COMPLETE | PRUNED | FAIL | Keresési epoch | Epoch trialonként | Trialidő (s) | s/epoch | Legjobb keresési val MSE (trial) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | európai call TPE | 25/25 | 10 | 15 | 0 | 500 | 20.0 | 265.8 | 0.53 | 1.46e-6 (#15) |
| MLPPricer | amerikai put retune | 25/25 | 9 | 16 | 0 | 544 | 21.8 | 191.0 | 0.35 | 1.17e-6 (#19) |
| DeepMLPPricer | európai call TPE | 30/30 | 7 | 23 | 0 | 397 | 13.2 | 412.2 | 1.04 | 8.40e-6 (#19) |
| DeepMLPPricer | amerikai put retune | 30/30 | 7 | 23 | 0 | 315 | 10.5 | 177.7 | 0.56 | 7.22e-6 (#16) |
| DenseMLPPricer | európai call TPE | 30/30 | 11 | 19 | 0 | 498 | 16.6 | 428.9 | 0.86 | 5.79e-6 (#25) |
| DenseMLPPricer | amerikai put retune | 30/30 | 10 | 20 | 0 | 534 | 17.8 | 254.9 | 0.48 | 1.62e-5 (#16) |
| ResNetPricer | európai call TPE | 40/40 | 13 | 27 | 0 | 671 | 16.8 | 1003.8 | 1.50 | 1.04e-5 (#35) |
| ResNetPricer | amerikai put retune | 40/40 | 10 | 30 | 0 | 642 | 16.1 | 540.8 | 0.84 | 1.48e-5 (#10) |
| GELUResNetPricer | európai call TPE | 40/40 | 11 | 29 | 0 | 665 | 16.6 | 597.9 | 0.90 | 4.65e-6 (#13) |
| GELUResNetPricer | amerikai put retune | 40/40 | 11 | 29 | 1 | 719 | 18.0 | 635.9 | 0.88 | 4.52e-6 (#14) |
| HighwayPricer | európai call TPE | 40/40 | 12 | 28 | 0 | 682 | 17.1 | 800.8 | 1.17 | 9.00e-6 (#38) |
| HighwayPricer | amerikai put retune | 40/40 | 13 | 27 | 0 | 682 | 17.1 | 758.5 | 1.11 | 7.39e-6 (#22) |
| FINNPricer | európai call TPE | 40/40 | 6 | 34 | 0 | 500 | 12.5 | 535.7 | 1.07 | 1.70e-5 (#19) |
| FINNPricer | amerikai put retune | 40/40 | 10 | 30 | 1 | 761 | 19.0 | 792.3 | 1.04 | 4.70e-6 (#22) |
| Összesen | európai call TPE |  | 70 | 175 | 0 | 3913 |  | 4044.9 |  |  |
| Összesen | amerikai put retune |  | 70 | 175 | 2 | 4197 |  | 3351.1 |  |  |

## FAIL trialok (amerikai put retune)

| Architektúra | Trial |
|---|---|
| GELUResNetPricer | #1 |
| FINNPricer | #1 |

## Hiperparaméter-fontosság (PED-ANOVA, tájékoztató)

| Architektúra | Paraméter | Európai call TPE | Amerikai put retune |
|---|---|---:|---:|
| MLPPricer | lr | 0.58 | 0.54 |
| MLPPricer | weight_decay | 0.42 | 0.46 |
| DeepMLPPricer | lr | 0.38 | 0.60 |
| DeepMLPPricer | weight_decay | 0.43 | 0.24 |
| DeepMLPPricer | dropout | 0.19 | 0.17 |
| DenseMLPPricer | lr | 0.07 | 0.30 |
| DenseMLPPricer | weight_decay | 0.41 | 0.37 |
| DenseMLPPricer | dropout | 0.52 | 0.33 |
| ResNetPricer | lr | 0.84 | 0.33 |
| ResNetPricer | weight_decay | 0.13 | 0.41 |
| ResNetPricer | dropout | 0.03 | 0.26 |
| GELUResNetPricer | lr | 0.02 | 0.09 |
| GELUResNetPricer | weight_decay | 0.94 | 0.81 |
| GELUResNetPricer | dropout | 0.05 | 0.10 |
| HighwayPricer | lr | 0.29 | 0.07 |
| HighwayPricer | weight_decay | 0.66 | 0.74 |
| HighwayPricer | dropout | 0.05 | 0.18 |
| FINNPricer | lr | 0.33 | 0.56 |
| FINNPricer | weight_decay | 0.39 | 0.20 |
| FINNPricer | dropout | 0.29 | 0.24 |

## A legfontosabb paraméter a két kampányban

| Architektúra | Európai call TPE | Amerikai put retune | Azonos |
|---|---|---|---:|
| MLPPricer | lr | lr | igen |
| DeepMLPPricer | weight_decay | lr | nem |
| DenseMLPPricer | dropout | weight_decay | nem |
| ResNetPricer | lr | weight_decay | nem |
| GELUResNetPricer | weight_decay | weight_decay | igen |
| HighwayPricer | weight_decay | weight_decay | igen |
| FINNPricer | weight_decay | lr | nem |
| Összesen |  |  | 3/7 |

## Amerikai és európai teszt-RMSE aránya három szinten

| Architektúra | uniform p30 | transfer p30 | retune p30 |
|---|---:|---:|---:|
| MLPPricer | 1.03 | 0.99 | 0.95 |
| DeepMLPPricer | 0.60 | 0.94 | 1.09 |
| DenseMLPPricer | 1.01 | 1.06 | 1.00 |
| ResNetPricer | 1.08 | 1.00 | 0.88 |
| GELUResNetPricer | 1.03 | 1.22 | 1.07 |
| HighwayPricer | 0.92 | 1.00 | 1.05 |
| FINNPricer | 1.06 | 1.04 | 1.08 |
| Medián | 1.03 | 1.00 | 1.05 |

## Rangsor teszt-RMSE szerint

| Hely | amerikai put, uniform p30 | amerikai put, transfer p30 | amerikai put, retune p30 | európai put, uniform p30 | európai put, transfer p30 | európai put, retune p30 | európai call, TPE p30 |
|---:|---|---|---|---|---|---|---|
| 1. | ResNetPricer (2.51e-3) | MLPPricer (7.83e-4) | MLPPricer (7.47e-4) | ResNetPricer (2.33e-3) | MLPPricer (7.91e-4) | MLPPricer (7.87e-4) | MLPPricer (9.78e-4) |
| 2. | MLPPricer (4.60e-3) | FINNPricer (1.96e-3) | FINNPricer (1.18e-3) | MLPPricer (4.45e-3) | GELUResNetPricer (1.87e-3) | FINNPricer (1.10e-3) | DenseMLPPricer (2.03e-3) |
| 3. | DeepMLPPricer (7.15e-3) | ResNetPricer (2.13e-3) | GELUResNetPricer (1.85e-3) | FINNPricer (9.19e-3) | FINNPricer (1.89e-3) | GELUResNetPricer (1.73e-3) | ResNetPricer (2.04e-3) |
| 4. | GELUResNetPricer (9.49e-3) | DeepMLPPricer (2.23e-3) | ResNetPricer (2.29e-3) | GELUResNetPricer (9.19e-3) | ResNetPricer (2.13e-3) | DeepMLPPricer (2.30e-3) | DeepMLPPricer (2.14e-3) |
| 5. | FINNPricer (9.78e-3) | GELUResNetPricer (2.28e-3) | DeepMLPPricer (2.50e-3) | DeepMLPPricer (1.20e-2) | DenseMLPPricer (2.23e-3) | HighwayPricer (2.50e-3) | HighwayPricer (2.94e-3) |
| 6. | HighwayPricer (1.27e-2) | DenseMLPPricer (2.37e-3) | HighwayPricer (2.62e-3) | HighwayPricer (1.38e-2) | DeepMLPPricer (2.37e-3) | ResNetPricer (2.60e-3) | FINNPricer (2.98e-3) |
| 7. | DenseMLPPricer (1.58e-2) | HighwayPricer (2.92e-3) | DenseMLPPricer (3.86e-3) | DenseMLPPricer (1.56e-2) | HighwayPricer (2.92e-3) | DenseMLPPricer (3.86e-3) | GELUResNetPricer (3.88e-3) |

## Kendall τ a rangsorok között

| Metrika | Egyik rangsor | Másik rangsor | Kendall τ | p |
|---|---|---|---:|---:|
| teszt-RMSE | amerikai put, uniform p30 | európai put, uniform p30 | 0.71 | 0.030 |
| teszt-RMSE | amerikai put, transfer p30 | európai put, transfer p30 | 0.62 | 0.069 |
| teszt-RMSE | amerikai put, retune p30 | európai put, retune p30 | 0.81 | 0.011 |
| teszt-RMSE | amerikai put, retune p30 | európai call, TPE p30 | −0.05 | 1.000 |
| teszt-RMSE | európai put, retune p30 | európai call, TPE p30 | −0.24 | 0.562 |
| teszt-RMSE | amerikai put, retune p30 | amerikai put, transfer p30 | 0.71 | 0.030 |
| teszt-RMSE | amerikai put, retune p30 | amerikai put, uniform p30 | 0.43 | 0.239 |
| val MSE | amerikai put, uniform p30 | európai put, uniform p30 | 0.71 | 0.030 |
| val MSE | amerikai put, transfer p30 | európai put, transfer p30 | 0.62 | 0.069 |
| val MSE | amerikai put, retune p30 | európai put, retune p30 | 0.81 | 0.011 |
| val MSE | amerikai put, retune p30 | európai call, TPE p30 | 0.05 | 1.000 |
| val MSE | európai put, retune p30 | európai call, TPE p30 | −0.14 | 0.773 |
| val MSE | amerikai put, retune p30 | amerikai put, transfer p30 | 0.71 | 0.030 |
| val MSE | amerikai put, retune p30 | amerikai put, uniform p30 | 0.43 | 0.239 |

## Amerikai és európai teszt-RMSE aránya a korai lehívási prémium szerint

| Architektúra | Szint | ~0 | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | uniform p30 | 1.23 | 1.08 | 0.97 | 1.00 | 1.04 | 0.93 |
| MLPPricer | transfer p30 | 0.97 | 1.00 | 0.99 | 1.02 | 1.01 | 0.92 |
| MLPPricer | retune p30 | 1.05 | 0.97 | 0.92 | 0.91 | 0.92 | 0.93 |
| DeepMLPPricer | uniform p30 | 0.46 | 0.70 | 0.63 | 0.60 | 0.57 | 0.57 |
| DeepMLPPricer | transfer p30 | 1.00 | 1.04 | 0.93 | 0.94 | 0.87 | 0.90 |
| DeepMLPPricer | retune p30 | 0.84 | 0.95 | 0.96 | 1.05 | 1.17 | 1.34 |
| DenseMLPPricer | uniform p30 | 0.98 | 1.00 | 1.03 | 1.04 | 1.01 | 1.03 |
| DenseMLPPricer | transfer p30 | 1.04 | 1.01 | 0.98 | 1.01 | 1.09 | 1.25 |
| DenseMLPPricer | retune p30 | 1.22 | 0.98 | 0.99 | 1.02 | 0.98 | 0.92 |
| ResNetPricer | uniform p30 | 1.04 | 1.06 | 1.06 | 1.10 | 1.11 | 1.10 |
| ResNetPricer | transfer p30 | 0.85 | 0.99 | 1.09 | 1.00 | 1.01 | 1.01 |
| ResNetPricer | retune p30 | 1.01 | 0.93 | 0.89 | 0.84 | 0.78 | 0.85 |
| GELUResNetPricer | uniform p30 | 1.11 | 0.99 | 0.97 | 1.03 | 1.10 | 1.07 |
| GELUResNetPricer | transfer p30 | 1.41 | 1.16 | 1.23 | 1.22 | 1.21 | 1.20 |
| GELUResNetPricer | retune p30 | 1.12 | 0.99 | 1.07 | 1.08 | 1.09 | 1.16 |
| HighwayPricer | uniform p30 | 0.89 | 0.92 | 0.98 | 0.98 | 0.91 | 0.87 |
| HighwayPricer | transfer p30 | 1.00 | 0.93 | 0.98 | 1.03 | 1.03 | 1.12 |
| HighwayPricer | retune p30 | 1.14 | 0.98 | 0.96 | 1.05 | 1.06 | 1.14 |
| FINNPricer | uniform p30 | 1.01 | 1.02 | 1.03 | 1.07 | 1.12 | 1.13 |
| FINNPricer | transfer p30 | 1.07 | 1.02 | 0.90 | 0.99 | 1.12 | 1.15 |
| FINNPricer | retune p30 | 1.03 | 0.98 | 1.04 | 1.16 | 1.15 | 1.23 |

## A legjobb amerikai modell hibája és a CRR-címke bizonytalansága

| Modell | RMSE ($) | MAE ($) | Abszolút hiba mediánja ($) | Relatív hiba mediánja (ár ≥ 1% · K) | A medián és a 0.04% aránya | Relatív hiba < 0.04% |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer, retune p30 | 0.0755 | 0.0402 | 0.0219 | 0.168% | 4.2 | 14.9% |

## A CRR-címke bizonytalansága

| Forrás | Érték |
|---|---|
| CRR 500 és 1000 lépés relatív eltérése | legfeljebb ~0.04% |
| A CRR-fa európai árának eltérése a BS ártól, medián | 0.0016 $ |

## Az indító fázis (0–9. trial) egyezése az európai call TPE kampánnyal

| Architektúra | Azonos trial, sorszám szerint | Mind azonos | Azonos trial, FAIL nélkül | Mind azonos, FAIL nélkül |
|---|---:|---:|---:|---:|
| MLPPricer | 10/10 | igen | 10/10 | igen |
| DeepMLPPricer | 10/10 | igen | 10/10 | igen |
| DenseMLPPricer | 10/10 | igen | 10/10 | igen |
| ResNetPricer | 10/10 | igen | 10/10 | igen |
| GELUResNetPricer | 2/10 | nem | 10/10 | igen |
| HighwayPricer | 10/10 | igen | 10/10 | igen |
| FINNPricer | 2/10 | nem | 10/10 | igen |
| Összesen |  | 5/7 |  | 7/7 |

## Ellenőrzések

| Ellenőrzés | Eredmény |
|---|---:|
| Protokoll: keresési tér, sampler, pruner, keresési türelem, batch, epoch és költségvetés az európai kampánnyal azonos | igen |
| Végső checkpointok | 14 |
| Paraméterszám = kiindulási v1 = HPO p30 = uniform p30 = transfer p30 (14 futás) | igen |
| Az európai retune hiperparaméterei bitre az amerikai páréval, a study legjobb trialjáéval és a `best_params` értékével | igen |
| Checkpoint-visszatöltés: legnagyobb relatív eltérés a mentett val MSE-től | 1.23e-8 |
| A uniform és a transfer sorok bitre a forrásaikkal (14 + 14 sor) | igen |
| A `run_hpo` importja nem változtatta a `hpo/` mappát (208 és 208 bejegyzés) | igen |
| A 200 epochos korlátig futott retune tanítások | 4 (mlp_american_retune_p30, mlp_european_retune_p30, finn_american_retune_p30, finn_european_retune_p30) |

## Amerikai és európai ár a teszthalmazon (retune p30)

| Modell | Átlagos különbség | Átlagban amerikai ≥ európai | Amerikai < európai | A különbség RMSE-je a célhoz |
|---|---:|---:|---:|---:|
| MLPPricer | 5.32e-3 | igen | 14.35% | 7.38e-4 |
| DeepMLPPricer | 6.09e-3 | igen | 12.10% | 1.84e-3 |
| DenseMLPPricer | 4.79e-3 | igen | 32.19% | 2.00e-3 |
| ResNetPricer | 5.31e-3 | igen | 19.42% | 1.99e-3 |
| GELUResNetPricer | 4.89e-3 | igen | 29.93% | 9.47e-4 |
| HighwayPricer | 5.04e-3 | igen | 21.15% | 1.20e-3 |
| FINNPricer | 5.17e-3 | igen | 15.78% | 7.16e-4 |
| Célváltozó | 5.29e-3 | – | – | – |

## Futásidő

| Architektúra | Keresés (s) | Trialidő (s) | Keresési epoch | Végső tanítás, amerikai (s) | Végső tanítás, európai (s) |
|---|---:|---:|---:|---:|---:|
| MLPPricer | 191.8 | 191.0 | 544 | 1139.6 | 1138.1 |
| DeepMLPPricer | 178.8 | 177.7 | 315 | 946.2 | 518.6 |
| DenseMLPPricer | 256.0 | 254.9 | 534 | 647.4 | 629.3 |
| ResNetPricer | 542.3 | 540.8 | 642 | 1026.1 | 949.0 |
| GELUResNetPricer | 637.9 | 635.9 | 719 | 815.9 | 804.2 |
| HighwayPricer | 760.5 | 758.5 | 682 | 586.6 | 774.9 |
| FINNPricer | 794.2 | 792.3 | 761 | 1833.3 | 1812.2 |
| Összesen | 3361.4 | 3351.1 | 4197 | 6995.1 | 6626.2 |

## Teljes futásidő

| Végső tanítás epochja | Keresés és végső tanítás (s) | Keresés és végső tanítás (óra) |
|---:|---:|---:|
| 1890 | 16982.7 | 4.72 |

## Indítások

| Indítás | Kezdés | Architektúrák |
|---:|---|---|
| 1 | 2026-10-14 02:35:22 | MLPPricer, DeepMLPPricer, DenseMLPPricer, ResNetPricer, GELUResNetPricer, HighwayPricer, FINNPricer |
| 2 | 2026-10-14 03:10:38 | MLPPricer, DeepMLPPricer, DenseMLPPricer, ResNetPricer, GELUResNetPricer, HighwayPricer, FINNPricer |
| 3 | 2026-10-14 08:59:49 | GELUResNetPricer, HighwayPricer, FINNPricer |
| 4 | 2026-10-14 11:08:11 | FINNPricer |

## Folytatás félbehagyott study-ból

| Architektúra | Időpont | Érvényes trial | FAIL-re állított trial | Sampler-állapot visszatöltve | Utolsó trial a mentés előtt | Konzisztens |
|---|---|---:|---:|---|---:|---|
| MLPPricer | 2026-10-14 03:10:39 | 25 | – | igen | #24 | igen |
| GELUResNetPricer | 2026-10-14 08:59:50 | 1 | #1 | igen | #0 | igen |
| FINNPricer | 2026-10-14 11:08:12 | 1 | #1 | igen | #0 | igen |
