## Direkt és reziduális (R1) modell az amerikai áron

| Architektúra | Val MSE, direkt | Val MSE, R1 | Teszt-RMSE, direkt | Teszt-RMSE, R1 | Teszt-RMSE változása | R1 és direkt aránya | Teszt-RMSE ($), direkt | Teszt-RMSE ($), R1 | Max. hiba ($), direkt | Max. hiba ($), R1 | R1 legjobb epoch | R1 futott epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 5.25e-7 | 2.86e-7 | 7.47e-4 | 5.30e-4 | −29.1% | 0.709 | 0.0755 | 0.0617 | 3.98 | 1.70 | 123 | 153 |
| DeepMLPPricer | 6.25e-6 | 1.62e-7 | 2.50e-3 | 4.04e-4 | −83.8% | 0.162 | 0.3097 | 0.0502 | 4.75 | 1.30 | 197 | 200 |
| DenseMLPPricer | 1.46e-5 | 9.59e-6 | 3.86e-3 | 3.11e-3 | −19.5% | 0.805 | 0.3872 | 0.3903 | 8.18 | 7.73 | 14 | 44 |
| ResNetPricer | 5.10e-6 | 2.77e-6 | 2.29e-3 | 1.66e-3 | −27.6% | 0.724 | 0.2428 | 0.1934 | 7.70 | 4.35 | 21 | 51 |
| GELUResNetPricer | 3.37e-6 | 1.04e-6 | 1.85e-3 | 1.03e-3 | −44.6% | 0.554 | 0.1868 | 0.1144 | 4.25 | 2.31 | 153 | 183 |
| HighwayPricer | 6.72e-6 | 6.37e-6 | 2.62e-3 | 2.53e-3 | −3.3% | 0.967 | 0.2870 | 0.3168 | 5.88 | 6.38 | 13 | 43 |
| FINNPricer | 1.36e-6 | 1.31e-6 | 1.18e-3 | 1.15e-3 | −2.4% | 0.976 | 0.1265 | 0.1314 | 3.09 | 2.31 | 199 | 200 |

## Összesítés

| Mutató | Érték |
|---|---:|
| Kisebb R1 teszt-RMSE | 7/7 |
| A teszt-RMSE változásának mediánja | −27.6% |
| A legjobb R1 modell (val MSE az áron) | DeepMLPPricer |
| A legjobb R1 modell teszt-RMSE-je | 4.04e-4 |

## Az R1 és a direkt teszt-RMSE aránya szegmensenként (medián a 7 architektúrán)

| Csoport | Szegmens | n | Medián arány | Javult architektúra | Legjobb arány | Legrosszabb arány |
|---|---|---:|---:|---:|---:|---:|
| moneyness | ITM (S/K<0,9) | 59 996 | 1.046 | 3/7 | 0.172 | 1.335 |
| moneyness | ATM (0,9≤S/K<1,1) | 30 146 | 0.558 | 7/7 | 0.182 | 0.838 |
| moneyness | OTM (S/K≥1,1) | 59 858 | 0.413 | 7/7 | 0.099 | 0.594 |
| lejarat | rövid (T<0,25) | 18 488 | 0.326 | 7/7 | 0.136 | 0.439 |
| lejarat | közép (0,25≤T<1) | 56 469 | 0.682 | 7/7 | 0.135 | 0.961 |
| lejarat | hosszú (1≤T≤2) | 75 043 | 1.115 | 2/7 | 0.184 | 1.409 |
| premium | ~0 (<1e-06) | 6 968 | 0.279 | 7/7 | 0.060 | 0.551 |
| premium | Q1 [1.00e-06; 2.71e-04) | 28 607 | 0.294 | 7/7 | 0.071 | 0.402 |
| premium | Q2 [2.71e-04; 1.17e-03) | 28 606 | 0.535 | 7/7 | 0.105 | 0.720 |
| premium | Q3 [1.17e-03; 3.10e-03) | 28 606 | 0.625 | 7/7 | 0.115 | 0.846 |
| premium | Q4 [3.10e-03; 7.81e-03) | 28 606 | 0.780 | 6/7 | 0.163 | 1.067 |
| premium | Q5 [7.81e-03; max) | 28 607 | 1.615 | 2/7 | 0.235 | 1.812 |

## Konzisztencia-ellenőrzések (medián a 7 architektúrán)

| Halmaz | Ellenőrzés | τ | Megsértés, direkt | Megsértés, R1 | Max. ($), direkt | Max. ($), R1 | Kisebb R1 arány |
|---|---|---|---:|---:|---:|---:|---:|
| grid | 1. P ≥ max(K−S,0) | 0 | 17.21% | 14.08% | 1.623 | 2.476 | 7/7 |
| grid | 1. P ≥ max(K−S,0) | 1e-6 | 17.21% | 14.07% | 1.623 | 2.476 | 7/7 |
| grid | 1. P ≥ max(K−S,0) | RMSE | 11.22% | 5.97% | 1.623 | 2.476 | 4/7 |
| grid | 2. P_am ≥ P_BS | 0 | 30.97% | 22.91% | 1.643 | 0.374 | 6/7 |
| grid | 2. P_am ≥ P_BS | 1e-6 | 30.97% | 22.86% | 1.643 | 0.374 | 6/7 |
| grid | 2. P_am ≥ P_BS | RMSE | 10.67% | 1.47% | 1.643 | 0.451 | 6/7 |
| grid | 3. P ≤ K | 0 | 0.00% | 0.00% | – | – | 0/7 |
| grid | 3. P ≤ K | 1e-6 | 0.00% | 0.00% | – | – | 0/7 |
| grid | 3. P ≤ K | RMSE | 0.00% | 0.00% | – | – | 0/7 |
| grid | 4a. monoton csökken S-ben | 0 | 3.13% | 3.02% | 0.027 | 0.005 | 3/7 |
| grid | 4a. monoton csökken S-ben | 1e-6 | 3.10% | 3.00% | 0.027 | 0.006 | 4/7 |
| grid | 4a. monoton csökken S-ben | RMSE | 0.00% | 0.00% | – | – | 0/7 |
| grid | 4b. ∂P/∂S ≥ −1 | 0 | 4.86% | 8.26% | 0.105 | 0.171 | 0/7 |
| grid | 4b. ∂P/∂S ≥ −1 | 1e-6 | 4.85% | 8.11% | 0.105 | 0.171 | 0/7 |
| grid | 4b. ∂P/∂S ≥ −1 | RMSE | 0.00% | 0.48% | 0.232 | 0.193 | 0/7 |
| grid | 5. monoton nő T-ben | 0 | 13.07% | 11.85% | 0.156 | 0.197 | 6/7 |
| grid | 5. monoton nő T-ben | 1e-6 | 13.03% | 11.84% | 0.156 | 0.197 | 6/7 |
| grid | 5. monoton nő T-ben | RMSE | 0.00% | 0.31% | 0.195 | 0.197 | 0/7 |
| grid | 6. konvex S-ben | 0 | 15.99% | 7.72% | 0.022 | 0.015 | 7/7 |
| grid | 6. konvex S-ben | 1e-6 | 15.78% | 6.21% | 0.022 | 0.029 | 7/7 |
| grid | 6. konvex S-ben | RMSE | 0.00% | 0.00% | – | 0.078 | 0/7 |
| test | 1. P ≥ max(K−S,0) | 0 | 11.92% | 10.31% | 2.970 | 2.311 | 4/7 |
| test | 1. P ≥ max(K−S,0) | 1e-6 | 11.92% | 10.30% | 2.970 | 2.311 | 4/7 |
| test | 1. P ≥ max(K−S,0) | RMSE | 4.77% | 4.66% | 2.970 | 2.311 | 4/7 |
| test | 2. P_am ≥ P_BS | 0 | 24.68% | 14.20% | 2.931 | 0.496 | 6/7 |
| test | 2. P_am ≥ P_BS | 1e-6 | 24.66% | 14.17% | 2.931 | 0.496 | 6/7 |
| test | 2. P_am ≥ P_BS | RMSE | 4.54% | 0.33% | 2.931 | 0.470 | 7/7 |
| test | 3. P ≤ K | 0 | 0.00% | 0.00% | – | – | 0/7 |
| test | 3. P ≤ K | 1e-6 | 0.00% | 0.00% | – | – | 0/7 |
| test | 3. P ≤ K | RMSE | 0.00% | 0.00% | – | – | 0/7 |
| test | 4a. monoton csökken S-ben | 0 | 1.00% | 1.52% | 0.030 | 0.005 | 3/7 |
| test | 4a. monoton csökken S-ben | 1e-6 | 0.98% | 1.24% | 0.030 | 0.006 | 4/7 |
| test | 4a. monoton csökken S-ben | RMSE | 0.00% | 0.00% | – | – | 0/7 |
| test | 4b. ∂P/∂S ≥ −1 | 0 | 3.62% | 6.23% | 0.162 | 0.311 | 1/7 |
| test | 4b. ∂P/∂S ≥ −1 | 1e-6 | 3.60% | 6.19% | 0.162 | 0.311 | 1/7 |
| test | 4b. ∂P/∂S ≥ −1 | RMSE | 0.00% | 0.04% | 0.251 | 0.302 | 0/7 |
| test | 5. monoton nő T-ben | 0 | 6.75% | 9.01% | 0.042 | 0.094 | 1/7 |
| test | 5. monoton nő T-ben | 1e-6 | 6.53% | 8.77% | 0.042 | 0.094 | 1/7 |
| test | 5. monoton nő T-ben | RMSE | 0.00% | 0.00% | – | – | 0/7 |
| test | 6. konvex S-ben | 0 | 10.54% | 5.30% | 0.025 | 0.027 | 7/7 |
| test | 6. konvex S-ben | 1e-6 | 10.28% | 4.30% | 0.025 | 0.040 | 7/7 |
| test | 6. konvex S-ben | RMSE | 0.00% | 0.00% | – | 0.175 | 0/7 |

## A P_am ≥ P_BS megsértésének felbontása (teszthalmaz)

| Architektúra | Modell | τ | Megsértés | A cél is sérti | Csak a modell sérti | Csak a modell, max. (ár/K) |
|---|---|---|---:|---:|---:|---:|
| MLPPricer | R1 | 0 | 3.39% | 3 812 | 1 266 | 2.77e-5 |
| MLPPricer | R1 | 1e-6 | 3.17% | 580 | 4 179 | 2.77e-5 |
| MLPPricer | R1 | RMSE | 0.00% | 0 | 0 | – |
| MLPPricer | direkt | 0 | 12.93% | 3 855 | 15 546 | 7.37e-3 |
| MLPPricer | direkt | 1e-6 | 12.90% | 1 141 | 18 209 | 7.37e-3 |
| MLPPricer | direkt | RMSE | 1.93% | 0 | 2 897 | 7.37e-3 |
| DeepMLPPricer | R1 | 0 | 3.63% | 2 500 | 2 940 | 1.09e-3 |
| DeepMLPPricer | R1 | 1e-6 | 3.58% | 639 | 4 729 | 1.09e-3 |
| DeepMLPPricer | R1 | RMSE | 0.17% | 0 | 251 | 1.09e-3 |
| DeepMLPPricer | direkt | 0 | 25.71% | 5 114 | 33 449 | 1.67e-2 |
| DeepMLPPricer | direkt | 1e-6 | 25.70% | 1 648 | 36 898 | 1.67e-2 |
| DeepMLPPricer | direkt | RMSE | 4.41% | 0 | 6 611 | 1.67e-2 |
| DenseMLPPricer | R1 | 0 | 19.49% | 7 142 | 22 099 | 3.37e-3 |
| DenseMLPPricer | R1 | 1e-6 | 19.48% | 2 862 | 26 352 | 3.37e-3 |
| DenseMLPPricer | R1 | RMSE | 0.00% | 0 | 4 | 3.37e-3 |
| DenseMLPPricer | direkt | 0 | 29.08% | 5 660 | 37 964 | 2.04e-2 |
| DenseMLPPricer | direkt | 1e-6 | 29.08% | 1 466 | 42 147 | 2.04e-2 |
| DenseMLPPricer | direkt | RMSE | 7.23% | 0 | 10 842 | 2.04e-2 |
| ResNetPricer | R1 | 0 | 31.45% | 6 872 | 40 301 | 3.11e-3 |
| ResNetPricer | R1 | 1e-6 | 31.43% | 2 995 | 44 152 | 3.36e-3 |
| ResNetPricer | R1 | RMSE | 3.37% | 0 | 5 061 | 3.36e-3 |
| ResNetPricer | direkt | 0 | 21.68% | 2 012 | 30 502 | 1.54e-2 |
| ResNetPricer | direkt | 1e-6 | 21.66% | 1 068 | 31 418 | 1.54e-2 |
| ResNetPricer | direkt | RMSE | 4.54% | 0 | 6 807 | 1.54e-2 |
| GELUResNetPricer | R1 | 0 | 10.00% | 3 448 | 11 550 | 1.03e-2 |
| GELUResNetPricer | R1 | 1e-6 | 9.97% | 1 031 | 13 919 | 1.03e-2 |
| GELUResNetPricer | R1 | RMSE | 1.40% | 0 | 2 107 | 1.03e-2 |
| GELUResNetPricer | direkt | 0 | 28.50% | 6 724 | 36 032 | 1.01e-2 |
| GELUResNetPricer | direkt | 1e-6 | 28.49% | 2 375 | 40 361 | 1.01e-2 |
| GELUResNetPricer | direkt | RMSE | 8.79% | 0 | 13 180 | 1.01e-2 |
| HighwayPricer | R1 | 0 | 15.15% | 4 342 | 18 389 | 4.89e-3 |
| HighwayPricer | R1 | 1e-6 | 15.13% | 1 556 | 21 144 | 4.89e-3 |
| HighwayPricer | R1 | RMSE | 0.33% | 0 | 497 | 4.89e-3 |
| HighwayPricer | direkt | 0 | 24.68% | 5 994 | 31 025 | 1.26e-2 |
| HighwayPricer | direkt | 1e-6 | 24.66% | 1 713 | 35 281 | 1.62e-2 |
| HighwayPricer | direkt | RMSE | 5.74% | 0 | 8 608 | 1.62e-2 |
| FINNPricer | R1 | 0 | 14.20% | 3 581 | 17 712 | 6.52e-3 |
| FINNPricer | R1 | 1e-6 | 14.17% | 1 110 | 20 138 | 6.52e-3 |
| FINNPricer | R1 | RMSE | 0.83% | 0 | 1 238 | 6.52e-3 |
| FINNPricer | direkt | 0 | 14.85% | 5 353 | 16 926 | 1.22e-2 |
| FINNPricer | direkt | 1e-6 | 14.84% | 1 350 | 20 903 | 1.22e-2 |
| FINNPricer | direkt | RMSE | 3.77% | 0 | 5 657 | 1.22e-2 |

## Korai leállás és plató

| Architektúra | lr | dropout | R1 legjobb epoch | R1 futott epoch | LR-felezés a legjobb epochig (R1) | Direkt legjobb epoch | Direkt futott epoch | R1 val MSE / a reziduum varianciája | R1 teszt R² a reziduumon | A direkt elérte az R1 legjobb val MSE-jét |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 2.73e-3 | – | 123 | 153 | 8 | 199 | 200 | 0.0031 | 0.9969 | nem |
| DeepMLPPricer | 1.72e-4 | 0.016 | 197 | 200 | 8 | 126 | 156 | 0.0018 | 0.9982 | nem |
| DenseMLPPricer | 1.90e-3 | 0.162 | 14 | 44 | 0 | 80 | 110 | 0.1056 | 0.8924 | nem |
| ResNetPricer | 8.60e-3 | 0.104 | 21 | 51 | 0 | 136 | 166 | 0.0305 | 0.9694 | nem |
| GELUResNetPricer | 1.66e-4 | 0.069 | 153 | 183 | 4 | 58 | 88 | 0.0114 | 0.9882 | nem |
| HighwayPricer | 2.39e-3 | 0.128 | 13 | 43 | 0 | 33 | 63 | 0.0702 | 0.9286 | nem |
| FINNPricer | 2.36e-4 | 0.039 | 199 | 200 | 5 | 193 | 200 | 0.0145 | 0.9852 | nem |

## Hibatérkép: a legjobb R1 és a legjobb direkt modell

| Mutató | Érték |
|---|---:|
| R1 modell | deep_mlp_residual_retune_p30 |
| Direkt modell | mlp_american_retune_p30 |
| Cellák, ahol az R1 jobb (400 cellából) | 86.8% |
| A cellánkénti log₂-arány mediánja | −0.832 |
| Spearman ρ (log₂-arány és prémium) | 0.887 |

## Hibatérkép: teszt-RMSE régiónként

| Régió | n | Teszt-RMSE, direkt | Teszt-RMSE, R1 | R1 és direkt aránya |
|---|---:|---:|---:|---:|
| OTM (m≥1,1) | 59 858 | 5.79e-4 | 1.74e-4 | 0.301 |
| ATM | 30 146 | 1.13e-3 | 4.27e-4 | 0.379 |
| ITM (m<0,9) | 59 996 | 6.50e-4 | 5.35e-4 | 0.823 |
| rövid T (<0,25) | 18 488 | 1.56e-3 | 4.17e-4 | 0.267 |
| ITM × hosszú T | 29 746 | 5.86e-4 | 5.97e-4 | 1.020 |
| lehívási sorok | 16 787 | 6.74e-4 | 8.89e-4 | 1.321 |
| prémium < 1e-6 | 6 968 | 8.95e-4 | 1.53e-4 | 0.171 |
| összes | 150 000 | 7.47e-4 | 4.04e-4 | 0.541 |

## Sebesség: a teljes út mediánja

| Architektúra | Eszköz | Batch | Direkt (s) | R1, BS NumPy-jal (s) | R1 és direkt aránya | R1, BS a GPU-n (s) | R1 GPU-BS-sel és direkt aránya |
|---|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | GPU | 1 | 5.53e-4 | 7.46e-4 | 1.35 | 9.14e-4 | 1.65 |
| MLPPricer | GPU | 1 000 | 5.65e-4 | 8.80e-4 | 1.56 | 9.35e-4 | 1.66 |
| MLPPricer | GPU | 100 000 | 4.67e-3 | 1.61e-2 | 3.46 | 6.78e-3 | 1.45 |
| MLPPricer | GPU | 1 000 000 | 6.67e-2 | 2.46e-1 | 3.68 | 7.66e-2 | 1.15 |
| MLPPricer | CPU | 1 | 1.36e-4 | 2.25e-4 | 1.65 | – | – |
| MLPPricer | CPU | 1 000 | 4.73e-4 | 7.55e-4 | 1.60 | – | – |
| DeepMLPPricer | GPU | 1 | 7.23e-4 | 1.12e-3 | 1.55 | 1.47e-3 | 2.03 |
| DeepMLPPricer | GPU | 1 000 | 6.89e-4 | 9.37e-4 | 1.36 | 1.20e-3 | 1.74 |
| DeepMLPPricer | GPU | 100 000 | 1.75e-2 | 2.82e-2 | 1.61 | 2.01e-2 | 1.15 |
| DeepMLPPricer | GPU | 1 000 000 | 1.94e-1 | 3.41e-1 | 1.76 | 2.03e-1 | 1.05 |
| DeepMLPPricer | CPU | 1 | 2.81e-4 | 4.49e-4 | 1.60 | – | – |
| DeepMLPPricer | CPU | 1 000 | 1.77e-3 | 2.10e-3 | 1.19 | – | – |
| DenseMLPPricer | GPU | 1 | 6.25e-4 | 9.58e-4 | 1.53 | 1.33e-3 | 2.13 |
| DenseMLPPricer | GPU | 1 000 | 6.18e-4 | 8.57e-4 | 1.39 | 1.10e-3 | 1.77 |
| DenseMLPPricer | GPU | 100 000 | 1.29e-2 | 2.40e-2 | 1.85 | 1.50e-2 | 1.16 |
| DenseMLPPricer | GPU | 1 000 000 | 1.47e-1 | 2.98e-1 | 2.02 | 1.63e-1 | 1.10 |
| DenseMLPPricer | CPU | 1 | 3.10e-4 | 4.80e-4 | 1.55 | – | – |
| DenseMLPPricer | CPU | 1 000 | 1.50e-3 | 1.88e-3 | 1.25 | – | – |
| ResNetPricer | GPU | 1 | 1.20e-3 | 1.73e-3 | 1.44 | 2.12e-3 | 1.77 |
| ResNetPricer | GPU | 1 000 | 1.14e-3 | 1.39e-3 | 1.22 | 1.64e-3 | 1.44 |
| ResNetPricer | GPU | 100 000 | 2.59e-2 | 3.73e-2 | 1.44 | 2.82e-2 | 1.09 |
| ResNetPricer | GPU | 1 000 000 | 2.76e-1 | 4.17e-1 | 1.51 | 2.93e-1 | 1.06 |
| ResNetPricer | CPU | 1 | 4.05e-4 | 5.92e-4 | 1.46 | – | – |
| ResNetPricer | CPU | 1 000 | 2.78e-3 | 3.09e-3 | 1.11 | – | – |
| GELUResNetPricer | GPU | 1 | 8.96e-4 | 1.66e-3 | 1.86 | 1.74e-3 | 1.94 |
| GELUResNetPricer | GPU | 1 000 | 8.33e-4 | 1.06e-3 | 1.27 | 1.33e-3 | 1.60 |
| GELUResNetPricer | GPU | 100 000 | 2.44e-2 | 3.54e-2 | 1.45 | 2.58e-2 | 1.06 |
| GELUResNetPricer | GPU | 1 000 000 | 2.57e-1 | 3.95e-1 | 1.53 | 2.67e-1 | 1.04 |
| GELUResNetPricer | CPU | 1 | 4.48e-4 | 6.48e-4 | 1.45 | – | – |
| GELUResNetPricer | CPU | 1 000 | 3.43e-3 | 3.27e-3 | 0.95 | – | – |
| HighwayPricer | GPU | 1 | 1.15e-3 | 1.75e-3 | 1.52 | 2.07e-3 | 1.79 |
| HighwayPricer | GPU | 1 000 | 1.06e-3 | 1.30e-3 | 1.22 | 1.54e-3 | 1.45 |
| HighwayPricer | GPU | 100 000 | 4.27e-2 | 5.41e-2 | 1.27 | 4.47e-2 | 1.05 |
| HighwayPricer | GPU | 1 000 000 | 4.45e-1 | 5.82e-1 | 1.31 | 4.58e-1 | 1.03 |
| HighwayPricer | CPU | 1 | 5.47e-4 | 8.09e-4 | 1.48 | – | – |
| HighwayPricer | CPU | 1 000 | 4.06e-3 | 4.68e-3 | 1.15 | – | – |
| FINNPricer | GPU | 1 | 1.18e-3 | 1.64e-3 | 1.38 | 2.04e-3 | 1.72 |
| FINNPricer | GPU | 1 000 | 1.20e-3 | 1.27e-3 | 1.06 | 1.54e-3 | 1.29 |
| FINNPricer | GPU | 100 000 | 2.53e-2 | 3.63e-2 | 1.44 | 2.70e-2 | 1.07 |
| FINNPricer | GPU | 1 000 000 | 2.69e-1 | 4.07e-1 | 1.52 | 2.86e-1 | 1.07 |
| FINNPricer | CPU | 1 | 5.48e-4 | 7.45e-4 | 1.36 | – | – |
| FINNPricer | CPU | 1 000 | 3.23e-3 | 3.64e-3 | 1.13 | – | – |
