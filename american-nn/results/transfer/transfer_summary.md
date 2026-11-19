## Teszt-RMSE, val MSE és tanítás

| Architektúra | Cél | Szint | Val MSE | Teszt-RMSE | Teszt MAE | Teszt R² | Teszt max. hiba | Teszt-RMSE ($) | Legjobb epoch | Futott epoch | 200 epochos korlát |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | amerikai | uniform p30 | 2.06e-5 | 4.60e-3 | 2.82e-3 | 0.999135 | 9.22e-2 | 4.76e-1 | 15 | 45 | nem |
| MLPPricer | amerikai | transfer p30 | 5.82e-7 | 7.83e-4 | 4.70e-4 | 0.999975 | 3.57e-2 | 7.81e-2 | 198 | 200 | igen |
| MLPPricer | európai | uniform p30 | 1.94e-5 | 4.45e-3 | 2.71e-3 | 0.999149 | 9.07e-2 | 4.90e-1 | 123 | 153 | nem |
| MLPPricer | európai | transfer p30 | 5.98e-7 | 7.91e-4 | 4.84e-4 | 0.999973 | 3.59e-2 | 8.01e-2 | 200 | 200 | igen |
| DeepMLPPricer | amerikai | uniform p30 | 5.10e-5 | 7.15e-3 | 5.56e-3 | 0.997910 | 5.74e-2 | 8.66e-1 | 36 | 66 | nem |
| DeepMLPPricer | amerikai | transfer p30 | 4.95e-6 | 2.23e-3 | 1.66e-3 | 0.999796 | 3.99e-2 | 2.49e-1 | 74 | 104 | nem |
| DeepMLPPricer | európai | uniform p30 | 1.45e-4 | 1.20e-2 | 9.74e-3 | 0.993804 | 7.58e-2 | 1.54e0 | 45 | 75 | nem |
| DeepMLPPricer | európai | transfer p30 | 5.58e-6 | 2.37e-3 | 1.76e-3 | 0.999759 | 3.80e-2 | 2.68e-1 | 68 | 98 | nem |
| DenseMLPPricer | amerikai | uniform p30 | 2.47e-4 | 1.58e-2 | 1.15e-2 | 0.989793 | 1.28e-1 | 1.48e0 | 67 | 97 | nem |
| DenseMLPPricer | amerikai | transfer p30 | 5.53e-6 | 2.37e-3 | 1.62e-3 | 0.999771 | 4.08e-2 | 2.48e-1 | 75 | 105 | nem |
| DenseMLPPricer | európai | uniform p30 | 2.42e-4 | 1.56e-2 | 1.13e-2 | 0.989508 | 1.26e-1 | 1.49e0 | 95 | 125 | nem |
| DenseMLPPricer | európai | transfer p30 | 4.91e-6 | 2.23e-3 | 1.50e-3 | 0.999787 | 4.05e-2 | 2.31e-1 | 70 | 100 | nem |
| ResNetPricer | amerikai | uniform p30 | 6.15e-6 | 2.51e-3 | 1.87e-3 | 0.999742 | 5.68e-2 | 2.62e-1 | 188 | 200 | igen |
| ResNetPricer | amerikai | transfer p30 | 4.41e-6 | 2.13e-3 | 1.43e-3 | 0.999814 | 5.87e-2 | 2.15e-1 | 121 | 151 | nem |
| ResNetPricer | európai | uniform p30 | 5.31e-6 | 2.33e-3 | 1.74e-3 | 0.999766 | 5.51e-2 | 2.37e-1 | 137 | 167 | nem |
| ResNetPricer | európai | transfer p30 | 4.36e-6 | 2.13e-3 | 1.41e-3 | 0.999806 | 5.91e-2 | 2.20e-1 | 138 | 168 | nem |
| GELUResNetPricer | amerikai | uniform p30 | 8.90e-5 | 9.49e-3 | 6.37e-3 | 0.996311 | 9.31e-2 | 1.02e0 | 24 | 54 | nem |
| GELUResNetPricer | amerikai | transfer p30 | 5.08e-6 | 2.28e-3 | 1.57e-3 | 0.999787 | 4.00e-2 | 2.57e-1 | 46 | 76 | nem |
| GELUResNetPricer | európai | uniform p30 | 8.36e-5 | 9.19e-3 | 6.01e-3 | 0.996369 | 9.22e-2 | 9.84e-1 | 17 | 47 | nem |
| GELUResNetPricer | európai | transfer p30 | 3.42e-6 | 1.87e-3 | 1.30e-3 | 0.999849 | 3.65e-2 | 2.10e-1 | 52 | 82 | nem |
| HighwayPricer | amerikai | uniform p30 | 1.60e-4 | 1.27e-2 | 9.44e-3 | 0.993356 | 1.13e-1 | 1.29e0 | 64 | 94 | nem |
| HighwayPricer | amerikai | transfer p30 | 8.40e-6 | 2.92e-3 | 2.05e-3 | 0.999651 | 4.35e-2 | 2.91e-1 | 126 | 156 | nem |
| HighwayPricer | európai | uniform p30 | 1.90e-4 | 1.38e-2 | 1.00e-2 | 0.991781 | 1.15e-1 | 1.42e0 | 78 | 108 | nem |
| HighwayPricer | európai | transfer p30 | 8.37e-6 | 2.92e-3 | 1.94e-3 | 0.999633 | 4.80e-2 | 2.99e-1 | 103 | 133 | nem |
| FINNPricer | amerikai | uniform p30 | 9.45e-5 | 9.78e-3 | 6.59e-3 | 0.996086 | 9.14e-2 | 1.07e0 | 14 | 44 | nem |
| FINNPricer | amerikai | transfer p30 | 3.74e-6 | 1.96e-3 | 1.36e-3 | 0.999843 | 3.68e-2 | 1.97e-1 | 196 | 200 | igen |
| FINNPricer | európai | uniform p30 | 8.36e-5 | 9.19e-3 | 5.96e-3 | 0.996369 | 9.26e-2 | 1.00e0 | 15 | 45 | nem |
| FINNPricer | európai | transfer p30 | 3.50e-6 | 1.89e-3 | 1.30e-3 | 0.999846 | 3.77e-2 | 1.94e-1 | 200 | 200 | igen |

## A transfer p30 változása a uniform p30-hoz képest

| Architektúra | Cél | Teszt-RMSE, uniform p30 | Teszt-RMSE, transfer p30 | Teszt-RMSE változása | Uniform és transfer aránya | Val MSE változása |
|---|---|---:|---:|---:|---:|---:|
| MLPPricer | amerikai | 4.60e-3 | 7.83e-4 | −83.0% | 5.87 | −97.2% |
| MLPPricer | európai | 4.45e-3 | 7.91e-4 | −82.2% | 5.63 | −96.9% |
| DeepMLPPricer | amerikai | 7.15e-3 | 2.23e-3 | −68.8% | 3.20 | −90.3% |
| DeepMLPPricer | európai | 1.20e-2 | 2.37e-3 | −80.3% | 5.07 | −96.1% |
| DenseMLPPricer | amerikai | 1.58e-2 | 2.37e-3 | −85.0% | 6.67 | −97.8% |
| DenseMLPPricer | európai | 1.56e-2 | 2.23e-3 | −85.7% | 7.01 | −98.0% |
| ResNetPricer | amerikai | 2.51e-3 | 2.13e-3 | −15.2% | 1.18 | −28.3% |
| ResNetPricer | európai | 2.33e-3 | 2.13e-3 | −8.8% | 1.10 | −17.9% |
| GELUResNetPricer | amerikai | 9.49e-3 | 2.28e-3 | −76.0% | 4.16 | −94.3% |
| GELUResNetPricer | európai | 9.19e-3 | 1.87e-3 | −79.6% | 4.90 | −95.9% |
| HighwayPricer | amerikai | 1.27e-2 | 2.92e-3 | −77.1% | 4.37 | −94.8% |
| HighwayPricer | európai | 1.38e-2 | 2.92e-3 | −78.9% | 4.73 | −95.6% |
| FINNPricer | amerikai | 9.78e-3 | 1.96e-3 | −79.9% | 4.99 | −96.0% |
| FINNPricer | európai | 9.19e-3 | 1.89e-3 | −79.4% | 4.86 | −95.8% |
| Medián, amerikai |  |  |  | −77.1% | 4.37 |  |
| Medián, európai |  |  |  | −79.6% | 4.90 |  |

## Rontott-e az átvitel

| Cél | Romlott teszt-RMSE-jű architektúra |
|---|---|
| amerikai | nincs |
| európai | nincs |

## Amerikai és európai teszt-RMSE aránya

| Architektúra | Amerikai, uniform p30 | Európai, uniform p30 | Arány, uniform p30 | Amerikai, transfer p30 | Európai, transfer p30 | Arány, transfer p30 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 4.60e-3 | 4.45e-3 | 1.03 | 7.83e-4 | 7.91e-4 | 0.99 |
| DeepMLPPricer | 7.15e-3 | 1.20e-2 | 0.60 | 2.23e-3 | 2.37e-3 | 0.94 |
| DenseMLPPricer | 1.58e-2 | 1.56e-2 | 1.01 | 2.37e-3 | 2.23e-3 | 1.06 |
| ResNetPricer | 2.51e-3 | 2.33e-3 | 1.08 | 2.13e-3 | 2.13e-3 | 1.00 |
| GELUResNetPricer | 9.49e-3 | 9.19e-3 | 1.03 | 2.28e-3 | 1.87e-3 | 1.22 |
| HighwayPricer | 1.27e-2 | 1.38e-2 | 0.92 | 2.92e-3 | 2.92e-3 | 1.00 |
| FINNPricer | 9.78e-3 | 9.19e-3 | 1.06 | 1.96e-3 | 1.89e-3 | 1.04 |
| Medián |  |  | 1.03 |  |  | 1.00 |
| Minimum |  |  | 0.60 |  |  | 0.94 |
| Maximum |  |  | 1.08 |  |  | 1.22 |

## Rangsor teszt-RMSE szerint

| Hely | amerikai put, uniform p30 | európai put, uniform p30 | amerikai put, transfer p30 | európai put, transfer p30 | európai call, TPE p30 |
|---:|---|---|---|---|---|
| 1. | ResNetPricer (2.51e-3) | ResNetPricer (2.33e-3) | MLPPricer (7.83e-4) | MLPPricer (7.91e-4) | MLPPricer (9.78e-4) |
| 2. | MLPPricer (4.60e-3) | MLPPricer (4.45e-3) | FINNPricer (1.96e-3) | GELUResNetPricer (1.87e-3) | DenseMLPPricer (2.03e-3) |
| 3. | DeepMLPPricer (7.15e-3) | FINNPricer (9.19e-3) | ResNetPricer (2.13e-3) | FINNPricer (1.89e-3) | ResNetPricer (2.04e-3) |
| 4. | GELUResNetPricer (9.49e-3) | GELUResNetPricer (9.19e-3) | DeepMLPPricer (2.23e-3) | ResNetPricer (2.13e-3) | DeepMLPPricer (2.14e-3) |
| 5. | FINNPricer (9.78e-3) | DeepMLPPricer (1.20e-2) | GELUResNetPricer (2.28e-3) | DenseMLPPricer (2.23e-3) | HighwayPricer (2.94e-3) |
| 6. | HighwayPricer (1.27e-2) | HighwayPricer (1.38e-2) | DenseMLPPricer (2.37e-3) | DeepMLPPricer (2.37e-3) | FINNPricer (2.98e-3) |
| 7. | DenseMLPPricer (1.58e-2) | DenseMLPPricer (1.56e-2) | HighwayPricer (2.92e-3) | HighwayPricer (2.92e-3) | GELUResNetPricer (3.88e-3) |

## Kendall τ a rangsorok között

| Metrika | Egyik rangsor | Másik rangsor | Kendall τ | p |
|---|---|---|---:|---:|
| teszt-RMSE | amerikai put, uniform p30 | európai put, uniform p30 | 0.71 | 0.030 |
| teszt-RMSE | amerikai put, transfer p30 | európai put, transfer p30 | 0.62 | 0.069 |
| teszt-RMSE | amerikai put, transfer p30 | európai call, TPE p30 | 0.24 | 0.562 |
| teszt-RMSE | európai put, transfer p30 | európai call, TPE p30 | 0.05 | 1.000 |
| teszt-RMSE | amerikai put, uniform p30 | amerikai put, transfer p30 | 0.52 | 0.136 |
| val MSE | amerikai put, uniform p30 | európai put, uniform p30 | 0.71 | 0.030 |
| val MSE | amerikai put, transfer p30 | európai put, transfer p30 | 0.62 | 0.069 |
| val MSE | amerikai put, transfer p30 | európai call, TPE p30 | 0.33 | 0.381 |
| val MSE | európai put, transfer p30 | európai call, TPE p30 | 0.14 | 0.773 |
| val MSE | amerikai put, uniform p30 | amerikai put, transfer p30 | 0.52 | 0.136 |

## Amerikai és európai teszt-RMSE aránya moneyness és lejárat szerint

| Architektúra | Szint | ITM (S/K<0,9) | ATM (0,9≤S/K<1,1) | OTM (S/K≥1,1) | rövid (T<0,25) | közép (0,25≤T<1) | hosszú (1≤T≤2) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | uniform p30 | 0.93 | 1.06 | 1.15 | 1.06 | 1.01 | 1.01 |
| MLPPricer | transfer p30 | 0.95 | 1.00 | 1.02 | 0.99 | 0.96 | 1.01 |
| DeepMLPPricer | uniform p30 | 0.55 | 0.64 | 0.75 | 0.72 | 0.60 | 0.57 |
| DeepMLPPricer | transfer p30 | 0.90 | 0.97 | 0.99 | 0.99 | 0.98 | 0.90 |
| DenseMLPPricer | uniform p30 | 0.97 | 1.04 | 1.01 | 0.98 | 1.00 | 1.04 |
| DenseMLPPricer | transfer p30 | 1.11 | 1.04 | 1.03 | 0.98 | 1.08 | 1.16 |
| ResNetPricer | uniform p30 | 1.12 | 1.09 | 1.03 | 1.05 | 1.06 | 1.11 |
| ResNetPricer | transfer p30 | 0.98 | 1.03 | 0.98 | 0.94 | 1.05 | 1.05 |
| GELUResNetPricer | uniform p30 | 1.05 | 1.03 | 1.02 | 1.01 | 1.03 | 1.05 |
| GELUResNetPricer | transfer p30 | 1.18 | 1.18 | 1.32 | 1.17 | 1.23 | 1.25 |
| HighwayPricer | uniform p30 | 0.89 | 0.93 | 0.93 | 0.92 | 0.94 | 0.91 |
| HighwayPricer | transfer p30 | 1.01 | 0.99 | 1.00 | 0.91 | 1.00 | 1.11 |
| FINNPricer | uniform p30 | 1.05 | 1.12 | 1.01 | 1.00 | 1.09 | 1.13 |
| FINNPricer | transfer p30 | 1.01 | 0.99 | 1.08 | 0.96 | 1.08 | 1.11 |

## Amerikai és európai teszt-RMSE aránya a korai lehívási prémium szerint

| Architektúra | Szint | ~0 (<1e-06) | Q1 [1.00e-06; 2.71e-04) | Q2 [2.71e-04; 1.17e-03) | Q3 [1.17e-03; 3.10e-03) | Q4 [3.10e-03; 7.81e-03) | Q5 [7.81e-03; max) |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | uniform p30 | 1.23 | 1.08 | 0.97 | 1.00 | 1.04 | 0.93 |
| MLPPricer | transfer p30 | 0.97 | 1.00 | 0.99 | 1.02 | 1.01 | 0.92 |
| DeepMLPPricer | uniform p30 | 0.46 | 0.70 | 0.63 | 0.60 | 0.57 | 0.57 |
| DeepMLPPricer | transfer p30 | 1.00 | 1.04 | 0.93 | 0.94 | 0.87 | 0.90 |
| DenseMLPPricer | uniform p30 | 0.98 | 1.00 | 1.03 | 1.04 | 1.01 | 1.03 |
| DenseMLPPricer | transfer p30 | 1.04 | 1.01 | 0.98 | 1.01 | 1.09 | 1.25 |
| ResNetPricer | uniform p30 | 1.04 | 1.06 | 1.06 | 1.10 | 1.11 | 1.10 |
| ResNetPricer | transfer p30 | 0.85 | 0.99 | 1.09 | 1.00 | 1.01 | 1.01 |
| GELUResNetPricer | uniform p30 | 1.11 | 0.99 | 0.97 | 1.03 | 1.10 | 1.07 |
| GELUResNetPricer | transfer p30 | 1.41 | 1.16 | 1.23 | 1.22 | 1.21 | 1.20 |
| HighwayPricer | uniform p30 | 0.89 | 0.92 | 0.98 | 0.98 | 0.91 | 0.87 |
| HighwayPricer | transfer p30 | 1.00 | 0.93 | 0.98 | 1.03 | 1.03 | 1.12 |
| FINNPricer | uniform p30 | 1.01 | 1.02 | 1.03 | 1.07 | 1.12 | 1.13 |
| FINNPricer | transfer p30 | 1.07 | 1.02 | 0.90 | 0.99 | 1.12 | 1.15 |

## Amerikai és európai ár a teszthalmazon (transfer p30)

| Modell | Átlagos különbség | Átlagban amerikai ≥ európai | Amerikai < európai | A különbség RMSE-je a célhoz |
|---|---:|---:|---:|---:|
| MLPPricer | 5.25e-3 | igen | 15.22% | 6.93e-4 |
| DeepMLPPricer | 4.87e-3 | igen | 26.14% | 1.65e-3 |
| DenseMLPPricer | 5.29e-3 | igen | 16.88% | 1.10e-3 |
| ResNetPricer | 5.17e-3 | igen | 22.21% | 2.06e-3 |
| GELUResNetPricer | 4.08e-3 | igen | 32.34% | 1.83e-3 |
| HighwayPricer | 5.28e-3 | igen | 17.99% | 1.62e-3 |
| FINNPricer | 4.99e-3 | igen | 19.78% | 1.21e-3 |
| Célváltozó | 5.29e-3 | – | – | – |

## Ellenőrzések

| Ellenőrzés | Eredmény |
|---|---:|
| Az amerikai és az európai adathalmaz páros | igen |
| Paraméterszám = kiindulási v1 = HPO p30 = uniform p30 (14 futás) | igen |
| `lr`, `weight_decay`, `dropout`, batch, epoch, seed és türelem bitre a `best_params_*.json` értéke | igen |
| Checkpoint-visszatöltés: legnagyobb relatív eltérés a mentett val MSE-től | 2.38e-8 |
| A uniform p30 sorok bitre a `american-nn/results/uniform/comparison_table.csv` soraival (14 sor) | igen |
| A 200 epochos korlátig futott tanítások | 4 |

## Átvitt hiperparaméterek

| Architektúra | lr | weight_decay | dropout | Forrás | Call TPE p30 val MSE | Call TPE p30 teszt-RMSE |
|---|---:|---:|---:|---|---:|---:|
| MLPPricer | 1.39e-3 | 1.36e-6 | – | `hpo/results/p30/best_params_mlp.json` | 9.01e-7 | 9.78e-4 |
| DeepMLPPricer | 1.50e-3 | 1.38e-6 | 0.0180 | `hpo/results/p30/best_params_deep_mlp.json` | 4.56e-6 | 2.14e-3 |
| DenseMLPPricer | 3.05e-3 | 1.08e-6 | 0.0122 | `hpo/results/p30/best_params_dense_mlp.json` | 4.03e-6 | 2.03e-3 |
| ResNetPricer | 4.50e-3 | 2.90e-4 | 0.3253 | `hpo/results/p30/best_params_resnet.json` | 4.00e-6 | 2.04e-3 |
| GELUResNetPricer | 2.39e-4 | 1.10e-6 | 0.1222 | `hpo/results/p30/best_params_gelu_resnet.json` | 1.48e-5 | 3.88e-3 |
| HighwayPricer | 2.21e-3 | 1.02e-6 | 0.4063 | `hpo/results/p30/best_params_highway.json` | 8.53e-6 | 2.94e-3 |
| FINNPricer | 3.37e-3 | 1.07e-6 | 0.0270 | `hpo/results/p30/best_params_finn.json` | 8.67e-6 | 2.98e-3 |

## Futásidő (transfer p30)

| Tanítás | Epoch | Összes idő (s) | Összes idő (óra) | Átlagos epochidő (s) |
|---:|---:|---:|---:|---:|
| 14 | 1973 | 11744 | 3.26 | 5.93 |
