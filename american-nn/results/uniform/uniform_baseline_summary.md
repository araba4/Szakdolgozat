## Teszt-RMSE, val MSE és tanítás

| Architektúra | Cél | Türelem | Paraméter | Val MSE | Teszt-RMSE | Teszt R² | Teszt max. hiba | Teszt-RMSE ($) | Legjobb epoch | Futott epoch | 200 epochos korlát | Tanítási idő (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | amerikai | 10 | 30 901 | 2.06e-5 | 4.60e-3 | 0.999135 | 9.22e-2 | 4.76e-1 | 15 | 25 | nem | 143 |
| MLPPricer | amerikai | 30 | 30 901 | 2.06e-5 | 4.60e-3 | 0.999135 | 9.22e-2 | 4.76e-1 | 15 | 45 | nem | 256 |
| MLPPricer | európai | 10 | 30 901 | 2.20e-5 | 4.75e-3 | 0.999031 | 8.94e-2 | 5.21e-1 | 25 | 35 | nem | 205 |
| MLPPricer | európai | 30 | 30 901 | 1.94e-5 | 4.45e-3 | 0.999149 | 9.07e-2 | 4.90e-1 | 123 | 153 | nem | 909 |
| DeepMLPPricer | amerikai | 10 | 267 265 | 3.94e-4 | 1.98e-2 | 0.983909 | 8.72e-2 | 2.17e0 | 4 | 14 | nem | 89 |
| DeepMLPPricer | amerikai | 30 | 267 265 | 5.10e-5 | 7.15e-3 | 0.997910 | 5.74e-2 | 8.66e-1 | 36 | 66 | nem | 411 |
| DeepMLPPricer | európai | 10 | 267 265 | 3.07e-4 | 1.75e-2 | 0.986866 | 1.01e-1 | 1.98e0 | 2 | 12 | nem | 75 |
| DeepMLPPricer | európai | 30 | 267 265 | 1.45e-4 | 1.20e-2 | 0.993804 | 7.58e-2 | 1.54e0 | 45 | 75 | nem | 448 |
| DenseMLPPricer | amerikai | 10 | 101 381 | 2.47e-4 | 1.58e-2 | 0.989793 | 1.28e-1 | 1.48e0 | 67 | 77 | nem | 447 |
| DenseMLPPricer | amerikai | 30 | 101 381 | 2.47e-4 | 1.58e-2 | 0.989793 | 1.28e-1 | 1.48e0 | 67 | 97 | nem | 560 |
| DenseMLPPricer | európai | 10 | 101 381 | 2.42e-4 | 1.56e-2 | 0.989508 | 1.26e-1 | 1.49e0 | 95 | 105 | nem | 609 |
| DenseMLPPricer | európai | 30 | 101 381 | 2.42e-4 | 1.56e-2 | 0.989508 | 1.26e-1 | 1.49e0 | 95 | 125 | nem | 726 |
| ResNetPricer | amerikai | 10 | 399 873 | 1.28e-5 | 3.60e-3 | 0.999470 | 5.83e-2 | 3.97e-1 | 66 | 76 | nem | 457 |
| ResNetPricer | amerikai | 30 | 399 873 | 6.15e-6 | 2.51e-3 | 0.999742 | 5.68e-2 | 2.62e-1 | 188 | 200 | igen | 1204 |
| ResNetPricer | európai | 10 | 399 873 | 3.04e-5 | 5.55e-3 | 0.998678 | 6.91e-2 | 6.06e-1 | 39 | 49 | nem | 296 |
| ResNetPricer | európai | 30 | 399 873 | 5.31e-6 | 2.33e-3 | 0.999766 | 5.51e-2 | 2.37e-1 | 137 | 167 | nem | 1009 |
| GELUResNetPricer | amerikai | 10 | 398 337 | 8.90e-5 | 9.49e-3 | 0.996311 | 9.31e-2 | 1.02e0 | 24 | 34 | nem | 202 |
| GELUResNetPricer | amerikai | 30 | 398 337 | 8.90e-5 | 9.49e-3 | 0.996311 | 9.31e-2 | 1.02e0 | 24 | 54 | nem | 321 |
| GELUResNetPricer | európai | 10 | 398 337 | 8.36e-5 | 9.19e-3 | 0.996369 | 9.22e-2 | 9.84e-1 | 17 | 27 | nem | 161 |
| GELUResNetPricer | európai | 30 | 398 337 | 8.36e-5 | 9.19e-3 | 0.996369 | 9.22e-2 | 9.84e-1 | 17 | 47 | nem | 280 |
| HighwayPricer | amerikai | 10 | 527 873 | 1.60e-4 | 1.27e-2 | 0.993356 | 1.13e-1 | 1.29e0 | 64 | 74 | nem | 464 |
| HighwayPricer | amerikai | 30 | 527 873 | 1.60e-4 | 1.27e-2 | 0.993356 | 1.13e-1 | 1.29e0 | 64 | 94 | nem | 587 |
| HighwayPricer | európai | 10 | 527 873 | 1.90e-4 | 1.38e-2 | 0.991781 | 1.15e-1 | 1.42e0 | 78 | 88 | nem | 547 |
| HighwayPricer | európai | 30 | 527 873 | 1.90e-4 | 1.38e-2 | 0.991781 | 1.15e-1 | 1.42e0 | 78 | 108 | nem | 674 |
| FINNPricer | amerikai | 10 | 402 882 | 9.45e-5 | 9.78e-3 | 0.996086 | 9.14e-2 | 1.07e0 | 14 | 24 | nem | 145 |
| FINNPricer | amerikai | 30 | 402 882 | 9.45e-5 | 9.78e-3 | 0.996086 | 9.14e-2 | 1.07e0 | 14 | 44 | nem | 265 |
| FINNPricer | európai | 10 | 402 882 | 8.36e-5 | 9.19e-3 | 0.996369 | 9.26e-2 | 1.00e0 | 15 | 25 | nem | 151 |
| FINNPricer | európai | 30 | 402 882 | 8.36e-5 | 9.19e-3 | 0.996369 | 9.26e-2 | 1.00e0 | 15 | 45 | nem | 272 |

## Amerikai és európai teszt-RMSE aránya

| Architektúra | Amerikai, p10 | Európai, p10 | Arány, p10 | Amerikai, p30 | Európai, p30 | Arány, p30 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 4.60e-3 | 4.75e-3 | 0.97 | 4.60e-3 | 4.45e-3 | 1.03 |
| DeepMLPPricer | 1.98e-2 | 1.75e-2 | 1.13 | 7.15e-3 | 1.20e-2 | 0.60 |
| DenseMLPPricer | 1.58e-2 | 1.56e-2 | 1.01 | 1.58e-2 | 1.56e-2 | 1.01 |
| ResNetPricer | 3.60e-3 | 5.55e-3 | 0.65 | 2.51e-3 | 2.33e-3 | 1.08 |
| GELUResNetPricer | 9.49e-3 | 9.19e-3 | 1.03 | 9.49e-3 | 9.19e-3 | 1.03 |
| HighwayPricer | 1.27e-2 | 1.38e-2 | 0.92 | 1.27e-2 | 1.38e-2 | 0.92 |
| FINNPricer | 9.78e-3 | 9.19e-3 | 1.06 | 9.78e-3 | 9.19e-3 | 1.06 |
| Medián |  |  | 1.01 |  |  | 1.03 |
| Minimum |  |  | 0.65 |  |  | 0.60 |
| Maximum |  |  | 1.13 |  |  | 1.08 |

## Rangsor teszt-RMSE szerint

| Hely | Amerikai, p10 | Európai, p10 | Amerikai, p30 | Európai, p30 |
|---:|---|---|---|---|
| 1. | ResNetPricer (3.60e-3) | MLPPricer (4.75e-3) | ResNetPricer (2.51e-3) | ResNetPricer (2.33e-3) |
| 2. | MLPPricer (4.60e-3) | ResNetPricer (5.55e-3) | MLPPricer (4.60e-3) | MLPPricer (4.45e-3) |
| 3. | GELUResNetPricer (9.49e-3) | FINNPricer (9.19e-3) | DeepMLPPricer (7.15e-3) | FINNPricer (9.19e-3) |
| 4. | FINNPricer (9.78e-3) | GELUResNetPricer (9.19e-3) | GELUResNetPricer (9.49e-3) | GELUResNetPricer (9.19e-3) |
| 5. | HighwayPricer (1.27e-2) | HighwayPricer (1.38e-2) | FINNPricer (9.78e-3) | DeepMLPPricer (1.20e-2) |
| 6. | DenseMLPPricer (1.58e-2) | DenseMLPPricer (1.56e-2) | HighwayPricer (1.27e-2) | HighwayPricer (1.38e-2) |
| 7. | DeepMLPPricer (1.98e-2) | DeepMLPPricer (1.75e-2) | DenseMLPPricer (1.58e-2) | DenseMLPPricer (1.56e-2) |

## Az amerikai és az európai rangsor Spearman-korrelációja

| Türelem | Legjobb, amerikai | Legjobb, európai | Spearman ρ |
|---|---|---|---:|
| p10 | ResNetPricer | MLPPricer | 0.93 |
| p30 | ResNetPricer | ResNetPricer | 0.86 |

## Teszt-RMSE moneyness szerint (p30)

| Architektúra | Cél | ITM (S/K<0,9), n = 59 996 | ATM (0,9≤S/K<1,1), n = 30 146 | OTM (S/K≥1,1), n = 59 858 |
|---|---|---:|---:|---:|
| MLPPricer | amerikai | 4.13e-3 | 6.09e-3 | 4.14e-3 |
| MLPPricer | európai | 4.46e-3 | 5.77e-3 | 3.60e-3 |
| DeepMLPPricer | amerikai | 8.88e-3 | 5.60e-3 | 5.76e-3 |
| DeepMLPPricer | európai | 1.62e-2 | 8.76e-3 | 7.68e-3 |
| DenseMLPPricer | amerikai | 1.20e-2 | 1.92e-2 | 1.72e-2 |
| DenseMLPPricer | európai | 1.23e-2 | 1.84e-2 | 1.70e-2 |
| ResNetPricer | amerikai | 2.38e-3 | 3.10e-3 | 2.31e-3 |
| ResNetPricer | európai | 2.13e-3 | 2.85e-3 | 2.24e-3 |
| GELUResNetPricer | amerikai | 9.28e-3 | 1.23e-2 | 7.94e-3 |
| GELUResNetPricer | európai | 8.86e-3 | 1.20e-2 | 7.81e-3 |
| HighwayPricer | amerikai | 1.06e-2 | 1.84e-2 | 1.11e-2 |
| HighwayPricer | európai | 1.19e-2 | 1.97e-2 | 1.19e-2 |
| FINNPricer | amerikai | 9.66e-3 | 1.33e-2 | 7.57e-3 |
| FINNPricer | európai | 9.20e-3 | 1.19e-2 | 7.49e-3 |

## Teszt-RMSE lejárat szerint (p30)

| Architektúra | Cél | rövid (T<0,25), n = 18 488 | közép (0,25≤T<1), n = 56 469 | hosszú (1≤T≤2), n = 75 043 |
|---|---|---:|---:|---:|
| MLPPricer | amerikai | 9.49e-3 | 3.13e-3 | 3.56e-3 |
| MLPPricer | európai | 8.98e-3 | 3.10e-3 | 3.53e-3 |
| DeepMLPPricer | amerikai | 7.67e-3 | 6.99e-3 | 7.13e-3 |
| DeepMLPPricer | európai | 1.07e-2 | 1.17e-2 | 1.25e-2 |
| DenseMLPPricer | amerikai | 2.78e-2 | 1.22e-2 | 1.40e-2 |
| DenseMLPPricer | európai | 2.83e-2 | 1.22e-2 | 1.34e-2 |
| ResNetPricer | amerikai | 3.95e-3 | 2.25e-3 | 2.23e-3 |
| ResNetPricer | európai | 3.76e-3 | 2.11e-3 | 2.00e-3 |
| GELUResNetPricer | amerikai | 1.66e-2 | 6.74e-3 | 8.82e-3 |
| GELUResNetPricer | európai | 1.64e-2 | 6.52e-3 | 8.39e-3 |
| HighwayPricer | amerikai | 2.18e-2 | 1.10e-2 | 1.08e-2 |
| HighwayPricer | európai | 2.37e-2 | 1.17e-2 | 1.19e-2 |
| FINNPricer | amerikai | 1.72e-2 | 7.42e-3 | 8.79e-3 |
| FINNPricer | európai | 1.72e-2 | 6.83e-3 | 7.81e-3 |

## Teszt-RMSE a korai lehívási prémium szerint (p30)

| Architektúra | Cél | ~0 (<1e-06), n = 6 968 | Q1 [1.00e-06; 2.71e-04), n = 28 607 | Q2 [2.71e-04; 1.17e-03), n = 28 606 | Q3 [1.17e-03; 3.10e-03), n = 28 606 | Q4 [3.10e-03; 7.81e-03), n = 28 606 | Q5 [7.81e-03; max), n = 28 607 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | amerikai | 5.47e-3 | 6.94e-3 | 3.93e-3 | 3.51e-3 | 3.39e-3 | 4.01e-3 |
| MLPPricer | európai | 4.44e-3 | 6.42e-3 | 4.04e-3 | 3.51e-3 | 3.26e-3 | 4.31e-3 |
| DeepMLPPricer | amerikai | 3.75e-3 | 6.11e-3 | 5.88e-3 | 6.18e-3 | 7.39e-3 | 9.98e-3 |
| DeepMLPPricer | európai | 8.10e-3 | 8.78e-3 | 9.31e-3 | 1.04e-2 | 1.29e-2 | 1.74e-2 |
| DenseMLPPricer | amerikai | 3.22e-2 | 1.95e-2 | 1.31e-2 | 1.27e-2 | 1.25e-2 | 1.37e-2 |
| DenseMLPPricer | európai | 3.27e-2 | 1.95e-2 | 1.28e-2 | 1.21e-2 | 1.23e-2 | 1.33e-2 |
| ResNetPricer | amerikai | 3.48e-3 | 3.13e-3 | 2.38e-3 | 2.16e-3 | 2.10e-3 | 2.37e-3 |
| ResNetPricer | európai | 3.34e-3 | 2.96e-3 | 2.23e-3 | 1.95e-3 | 1.89e-3 | 2.16e-3 |
| GELUResNetPricer | amerikai | 1.41e-2 | 1.14e-2 | 8.04e-3 | 7.66e-3 | 7.90e-3 | 1.04e-2 |
| GELUResNetPricer | európai | 1.27e-2 | 1.15e-2 | 8.27e-3 | 7.45e-3 | 7.16e-3 | 9.79e-3 |
| HighwayPricer | amerikai | 2.22e-2 | 1.56e-2 | 1.12e-2 | 1.07e-2 | 9.89e-3 | 1.23e-2 |
| HighwayPricer | európai | 2.49e-2 | 1.69e-2 | 1.14e-2 | 1.09e-2 | 1.09e-2 | 1.41e-2 |
| FINNPricer | amerikai | 1.34e-2 | 1.15e-2 | 8.29e-3 | 8.05e-3 | 8.55e-3 | 1.09e-2 |
| FINNPricer | európai | 1.33e-2 | 1.13e-2 | 8.06e-3 | 7.56e-3 | 7.61e-3 | 9.66e-3 |

## Amerikai és európai teszt-RMSE aránya a korai lehívási prémium szerint (p30)

| Architektúra | ~0 | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 1.23 | 1.08 | 0.97 | 1.00 | 1.04 | 0.93 |
| DeepMLPPricer | 0.46 | 0.70 | 0.63 | 0.60 | 0.57 | 0.57 |
| DenseMLPPricer | 0.98 | 1.00 | 1.03 | 1.04 | 1.01 | 1.03 |
| ResNetPricer | 1.04 | 1.06 | 1.06 | 1.10 | 1.11 | 1.10 |
| GELUResNetPricer | 1.11 | 0.99 | 0.97 | 1.03 | 1.10 | 1.07 |
| HighwayPricer | 0.89 | 0.92 | 0.98 | 0.98 | 0.91 | 0.87 |
| FINNPricer | 1.01 | 1.02 | 1.03 | 1.07 | 1.12 | 1.13 |

## A p10 és a p30 tanítási görbéje a közös epochokon

| Futás | Közös epoch | Bitre azonos | Első eltérő epoch |
|---|---:|---:|---:|
| mlp_american | 25 | igen | – |
| mlp_european | 35 | igen | – |
| deep_mlp_american | 14 | igen | – |
| deep_mlp_european | 12 | igen | – |
| dense_mlp_american | 77 | igen | – |
| dense_mlp_european | 105 | igen | – |
| resnet_american | 76 | igen | – |
| resnet_european | 49 | igen | – |
| gelu_resnet_american | 34 | igen | – |
| gelu_resnet_european | 27 | igen | – |
| highway_american | 74 | igen | – |
| highway_european | 88 | igen | – |
| finn_american | 24 | igen | – |
| finn_european | 25 | igen | – |

## Amerikai és európai ár a teszthalmazon

| Modell | Átlagos amerikai ár/K | Átlagos európai ár/K | Átlagos különbség | Amerikai < európai | A különbség RMSE-je a célhoz |
|---|---:|---:|---:|---:|---:|
| MLPPricer, p10 | 0.213223 | 0.207586 | 5.64e-3 | 16.12% | 2.43e-3 |
| MLPPricer, p30 | 0.213223 | 0.207646 | 5.58e-3 | 20.01% | 2.71e-3 |
| DeepMLPPricer, p10 | 0.214088 | 0.211024 | 3.06e-3 | 49.76% | 8.85e-3 |
| DeepMLPPricer, p30 | 0.216347 | 0.212668 | 3.68e-3 | 31.09% | 7.61e-3 |
| DenseMLPPricer, p10 | 0.212617 | 0.207112 | 5.50e-3 | 17.98% | 3.75e-3 |
| DenseMLPPricer, p30 | 0.212617 | 0.207112 | 5.50e-3 | 17.98% | 3.75e-3 |
| ResNetPricer, p10 | 0.212659 | 0.207653 | 5.01e-3 | 21.75% | 5.57e-3 |
| ResNetPricer, p30 | 0.212692 | 0.207533 | 5.16e-3 | 21.69% | 1.19e-3 |
| GELUResNetPricer, p10 | 0.213231 | 0.208845 | 4.39e-3 | 36.38% | 3.05e-3 |
| GELUResNetPricer, p30 | 0.213231 | 0.208845 | 4.39e-3 | 36.38% | 3.05e-3 |
| HighwayPricer, p10 | 0.212596 | 0.208040 | 4.56e-3 | 33.60% | 4.68e-3 |
| HighwayPricer, p30 | 0.212596 | 0.208040 | 4.56e-3 | 33.60% | 4.68e-3 |
| FINNPricer, p10 | 0.213564 | 0.208291 | 5.27e-3 | 17.43% | 2.85e-3 |
| FINNPricer, p30 | 0.213564 | 0.208291 | 5.27e-3 | 17.43% | 2.85e-3 |
| Célváltozó | – | – | 5.29e-3 | 5.0213% | – |

## Ellenőrzések

| Ellenőrzés | Eredmény |
|---|---:|
| Az amerikai és az európai adathalmaz páros | igen |
| Checkpoint-visszatöltés: legnagyobb relatív eltérés a mentett val MSE-től | 1.41e-8 |
| A checkpointok célváltozója helyes | igen |
| Paraméterszám = kiindulási v1 (28 futás) | igen |
| A kiindulási `train_model` és a `train_loop` bitre azonos (dense_mlp, 3 epoch) | igen |

## Európai call, kiindulási v1 modellek (tájékoztató)

| Architektúra | Call v1 val MSE | Call v1 teszt-RMSE | Európai put p10 teszt-RMSE | Amerikai put p10 teszt-RMSE |
|---|---:|---:|---:|---:|
| MLPPricer | 2.06e-5 | 4.60e-3 | 4.75e-3 | 4.60e-3 |
| DeepMLPPricer | 3.63e-5 | 6.06e-3 | 1.75e-2 | 1.98e-2 |
| DenseMLPPricer | 3.23e-4 | 1.80e-2 | 1.56e-2 | 1.58e-2 |
| ResNetPricer | 3.85e-5 | 6.22e-3 | 5.55e-3 | 3.60e-3 |
| GELUResNetPricer | 8.50e-5 | 9.29e-3 | 9.19e-3 | 9.49e-3 |
| HighwayPricer | 2.20e-4 | 1.49e-2 | 1.38e-2 | 1.27e-2 |
| FINNPricer | 8.24e-5 | 9.14e-3 | 9.19e-3 | 9.78e-3 |

## Futásidő

| Tanítás | Összes idő (s) | Összes idő (óra) | Átlagos epochidő (s) |
|---:|---:|---:|---:|
| 28 | 11918 | 3.31 | 6.01 |
