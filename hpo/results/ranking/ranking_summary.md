## Európai call: teszt-RMSE szintenként (rang)

| Architektúra | Paraméter | kiindulási v1 | kiindulási legjobb | TPE p10 | TPE p30 | RS p30 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 4.60e-3 (1.) | 4.60e-3 (2.) | 9.78e-4 (1.) | 9.78e-4 (1.) | 9.05e-4 (1.) |
| DeepMLPPricer | 267 265 | 6.06e-3 (2.) | 6.06e-3 (3.) | 2.47e-3 (3.) | 2.14e-3 (4.) | 2.95e-3 (2.) |
| DenseMLPPricer | 101 381 | 1.80e-2 (7.) | 1.80e-2 (7.) | 2.04e-3 (2.) | 2.03e-3 (2.) | 4.37e-3 (7.) |
| ResNetPricer | 399 873 | 6.22e-3 (3.) | 2.91e-3 (1.) | 4.28e-3 (7.) | 2.04e-3 (3.) | 3.98e-3 (6.) |
| GELUResNetPricer | 398 337 | 9.29e-3 (5.) | 9.29e-3 (5.) | 3.88e-3 (6.) | 3.88e-3 (7.) | 3.97e-3 (5.) |
| HighwayPricer | 527 873 | 1.49e-2 (6.) | 1.49e-2 (6.) | 2.95e-3 (4.) | 2.94e-3 (5.) | 3.06e-3 (3.) |
| FINNPricer | 402 882 | 9.14e-3 (4.) | 9.14e-3 (4.) | 2.98e-3 (5.) | 2.98e-3 (6.) | 3.81e-3 (4.) |

## Európai call: val MSE szintenként (rang)

| Architektúra | Paraméter | kiindulási v1 | kiindulási legjobb | TPE p10 | TPE p30 | RS p30 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 2.06e-5 (1.) | 2.06e-5 (2.) | 9.01e-7 (1.) | 9.01e-7 (1.) | 7.73e-7 (1.) |
| DeepMLPPricer | 267 265 | 3.63e-5 (2.) | 3.63e-5 (3.) | 6.05e-6 (3.) | 4.56e-6 (4.) | 8.61e-6 (2.) |
| DenseMLPPricer | 101 381 | 3.23e-4 (7.) | 3.23e-4 (7.) | 4.06e-6 (2.) | 4.03e-6 (3.) | 1.86e-5 (7.) |
| ResNetPricer | 399 873 | 3.85e-5 (3.) | 8.35e-6 (1.) | 1.81e-5 (7.) | 4.00e-6 (2.) | 1.55e-5 (6.) |
| GELUResNetPricer | 398 337 | 8.50e-5 (5.) | 8.50e-5 (5.) | 1.48e-5 (6.) | 1.48e-5 (7.) | 1.54e-5 (5.) |
| HighwayPricer | 527 873 | 2.20e-4 (6.) | 2.20e-4 (6.) | 8.55e-6 (4.) | 8.53e-6 (5.) | 9.21e-6 (3.) |
| FINNPricer | 402 882 | 8.24e-5 (4.) | 8.24e-5 (4.) | 8.67e-6 (5.) | 8.67e-6 (6.) | 1.45e-5 (4.) |

## Európai call: Kendall τ a szintpárok között, teszt-RMSE

| | kiindulási v1 | kiindulási legjobb | TPE p10 | TPE p30 | RS p30 |
|---|---:|---:|---:|---:|---:|
| kiindulási v1 | 1 | 0.81 (p = 0.011) | 0.05 (p = 1.000) | 0.24 (p = 0.562) | 0.52 (p = 0.136) |
| kiindulási legjobb | 0.81 (p = 0.011) | 1 | -0.14 (p = 0.773) | 0.24 (p = 0.562) | 0.33 (p = 0.381) |
| TPE p10 | 0.05 (p = 1.000) | -0.14 (p = 0.773) | 1 | 0.62 (p = 0.069) | 0.52 (p = 0.136) |
| TPE p30 | 0.24 (p = 0.562) | 0.24 (p = 0.562) | 0.62 (p = 0.069) | 1 | 0.14 (p = 0.773) |
| RS p30 | 0.52 (p = 0.136) | 0.33 (p = 0.381) | 0.52 (p = 0.136) | 0.14 (p = 0.773) | 1 |

## Európai call: Kendall τ a szintpárok között, val MSE

| | kiindulási v1 | kiindulási legjobb | TPE p10 | TPE p30 | RS p30 |
|---|---:|---:|---:|---:|---:|
| kiindulási v1 | 1 | 0.81 (p = 0.011) | 0.05 (p = 1.000) | 0.33 (p = 0.381) | 0.52 (p = 0.136) |
| kiindulási legjobb | 0.81 (p = 0.011) | 1 | -0.14 (p = 0.773) | 0.33 (p = 0.381) | 0.33 (p = 0.381) |
| TPE p10 | 0.05 (p = 1.000) | -0.14 (p = 0.773) | 1 | 0.52 (p = 0.136) | 0.52 (p = 0.136) |
| TPE p30 | 0.33 (p = 0.381) | 0.33 (p = 0.381) | 0.52 (p = 0.136) | 1 | 0.24 (p = 0.562) |
| RS p30 | 0.52 (p = 0.136) | 0.33 (p = 0.381) | 0.52 (p = 0.136) | 0.24 (p = 0.562) | 1 |

## Európai call: Kendall τ a teszt-RMSE és a val MSE rangsora között, szintenként

| Szint | τ | p |
|---|---:|---:|
| kiindulási v1 | 1.00 | 0.000 |
| kiindulási legjobb | 1.00 | 0.000 |
| TPE p10 | 1.00 | 0.000 |
| TPE p30 | 0.90 | 0.003 |
| RS p30 | 1.00 | 0.000 |

## DenseMLP és Highway: a kiindulási v1 és a hangolt szintek

| Architektúra | Paraméter | Teszt-RMSE, kiindulási v1 (rang) | Teszt-RMSE, TPE p10 (rang) | Teszt-RMSE, TPE p30 (rang) | Teszt-RMSE, RS p30 (rang) | Teszt-RMSE változása, TPE p10 / TPE p30 / RS p30 | Val MSE változása, TPE p10 / TPE p30 / RS p30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseMLPPricer | 101 381 | 1.80e-2 (7.) | 2.04e-3 (2.) | 2.03e-3 (2.) | 4.37e-3 (7.) | −88.7% / −88.7% / −75.8% | −98.7% / −98.8% / −94.2% |
| HighwayPricer | 527 873 | 1.49e-2 (6.) | 2.95e-3 (4.) | 2.94e-3 (5.) | 3.06e-3 (3.) | −80.2% / −80.2% / −79.4% | −96.1% / −96.1% / −95.8% |

## Amerikai put: teszt-RMSE szintenként (rang)

| Architektúra | Paraméter | uniform p30 | transfer p30 | retune p30 | R1 (BS + NN) |
|---|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 4.60e-3 (2.) | 7.83e-4 (1.) | 7.47e-4 (1.) | 5.30e-4 (2.) |
| DeepMLPPricer | 267 265 | 7.15e-3 (3.) | 2.23e-3 (4.) | 2.50e-3 (5.) | 4.04e-4 (1.) |
| DenseMLPPricer | 101 381 | 1.58e-2 (7.) | 2.37e-3 (6.) | 3.86e-3 (7.) | 3.11e-3 (7.) |
| ResNetPricer | 399 873 | 2.51e-3 (1.) | 2.13e-3 (3.) | 2.29e-3 (4.) | 1.66e-3 (5.) |
| GELUResNetPricer | 398 337 | 9.49e-3 (4.) | 2.28e-3 (5.) | 1.85e-3 (3.) | 1.03e-3 (3.) |
| HighwayPricer | 527 873 | 1.27e-2 (6.) | 2.92e-3 (7.) | 2.62e-3 (6.) | 2.53e-3 (6.) |
| FINNPricer | 402 882 | 9.78e-3 (5.) | 1.96e-3 (2.) | 1.18e-3 (2.) | 1.15e-3 (4.) |

## Amerikai put: val MSE szintenként (rang)

| Architektúra | Paraméter | uniform p30 | transfer p30 | retune p30 | R1 (BS + NN) |
|---|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 2.06e-5 (2.) | 5.82e-7 (1.) | 5.25e-7 (1.) | 2.86e-7 (2.) |
| DeepMLPPricer | 267 265 | 5.10e-5 (3.) | 4.95e-6 (4.) | 6.25e-6 (5.) | 1.62e-7 (1.) |
| DenseMLPPricer | 101 381 | 2.47e-4 (7.) | 5.53e-6 (6.) | 1.46e-5 (7.) | 9.59e-6 (7.) |
| ResNetPricer | 399 873 | 6.15e-6 (1.) | 4.41e-6 (3.) | 5.10e-6 (4.) | 2.77e-6 (5.) |
| GELUResNetPricer | 398 337 | 8.90e-5 (4.) | 5.08e-6 (5.) | 3.37e-6 (3.) | 1.04e-6 (3.) |
| HighwayPricer | 527 873 | 1.60e-4 (6.) | 8.40e-6 (7.) | 6.72e-6 (6.) | 6.37e-6 (6.) |
| FINNPricer | 402 882 | 9.45e-5 (5.) | 3.74e-6 (2.) | 1.36e-6 (2.) | 1.31e-6 (4.) |

## Amerikai put: Kendall τ a szintpárok között, teszt-RMSE

| | uniform p30 | transfer p30 | retune p30 | R1 (BS + NN) |
|---|---:|---:|---:|---:|
| uniform p30 | 1 | 0.52 (p = 0.136) | 0.43 (p = 0.239) | 0.52 (p = 0.136) |
| transfer p30 | 0.52 (p = 0.136) | 1 | 0.71 (p = 0.030) | 0.43 (p = 0.239) |
| retune p30 | 0.43 (p = 0.239) | 0.71 (p = 0.030) | 1 | 0.52 (p = 0.136) |
| R1 (BS + NN) | 0.52 (p = 0.136) | 0.43 (p = 0.239) | 0.52 (p = 0.136) | 1 |

## Paraméterszám és teszt-RMSE

| Szint | Kendall τ | p | Spearman ρ | p |
|---|---:|---:|---:|---:|
| európai call, kiindulási v1 | 0.33 | 0.381 | 0.36 | 0.432 |
| európai call, kiindulási legjobb | 0.14 | 0.773 | 0.14 | 0.760 |
| európai call, TPE p10 | 0.52 | 0.136 | 0.68 | 0.094 |
| európai call, TPE p30 | 0.52 | 0.136 | 0.68 | 0.094 |
| európai call, RS p30 | 0.05 | 1.000 | 0.14 | 0.760 |
| amerikai put, uniform p30 | 0.24 | 0.562 | 0.21 | 0.645 |
| amerikai put, transfer p30 | 0.14 | 0.773 | 0.32 | 0.482 |
| amerikai put, retune p30 | 0.05 | 1.000 | 0.14 | 0.760 |
| amerikai put, R1 (BS + NN) | 0.33 | 0.381 | 0.36 | 0.432 |

## Verifikáció

| Ellenőrzés | Összevetés | Eltérés | Legnagyobb relatív eltérés |
|---|---:|---:|---:|
| Európai checkpointok (34): újraszámolt és mentett teszt-RMSE és val MSE | 158 | 0 | teszt-RMSE 0, val MSE 1.60e-7 |
| A kiindulási legjobb választása (v1 vagy v2) és az `onlab_best_ref` mező | 7 | 0 | – |
| `hpo/results/random/comparison/random_vs_tpe_table.csv` | 42 | 0 | 0 |
| Paraméterszám: `count_parameters` a 34 európai és a 28 amerikai checkpointon, a mentett értékekkel (architektúránként) | 7 | 0 | – |
| Amerikai rangok és az `american-nn/results/retune/ranking.csv` | 42 | 0 | – |
| TPE p30 rangok és a `ranking.csv` `european_call` sorai | 14 | 0 | – |
