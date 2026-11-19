## Hibatérkép: modellválasztás

| Szerep | Modell |
|---|---|
| Legjobb amerikai (legkisebb val MSE) | `mlp_american_retune_p30` |
| Legjobb európai put (legkisebb val MSE) | `mlp_european_retune_p30` |
| Kiegészítő pár | retune p30 ResNetPricer, mindkét célon |
| Cellák | 20×20 egyenletes (m, T); σ-szeletben 10×10 |

## Hibatérkép: a cellákból visszaszámolt teszt-RMSE

| Modell | Cellákból | levels_table.csv | Relatív eltérés |
|---|---:|---:|---:|
| `mlp_american_retune_p30` | 7.47262e-4 | 7.47262e-4 | 1.47e-8 |
| `mlp_european_retune_p30` | 7.86518e-4 | 7.86518e-4 | 5.40e-10 |
| `resnet_american_retune_p30` | 2.28828e-3 | 2.28828e-3 | 4.26e-8 |
| `resnet_european_retune_p30` | 2.59535e-3 | 2.59535e-3 | 1.23e-8 |

## Hibatérkép: teszt-RMSE régiónként

| Modell | Spearman ρ (cella-RMSE és prémium) | ITM × hosszú T | Máshol | Lehívási sorok | Folytatási sorok | Legrosszabb cella (m; T) | A legrosszabb cella RMSE-je |
|---|---:|---:|---:|---:|---:|---|---:|
| `mlp_american_retune_p30` | 0.235 | 5.86e-4 | 7.82e-4 | 6.74e-4 | 7.56e-4 | [1.05; 1.10) × [5.00e-3; 0.105) | 4.53e-3 |
| `mlp_european_retune_p30` | 0.184 | 6.28e-4 | 8.21e-4 | 7.97e-4 | 7.85e-4 | [1.00; 1.05) × [5.00e-3; 0.105) | 4.68e-3 |
| `resnet_american_retune_p30` | 0.235 | 2.15e-3 | 2.32e-3 | 2.60e-3 | 2.25e-3 | [1.05; 1.10) × [5.00e-3; 0.105) | 0.0123 |
| `resnet_european_retune_p30` | 0.310 | 2.56e-3 | 2.60e-3 | 2.85e-3 | 2.56e-3 | [1.05; 1.10) × [5.00e-3; 0.105) | 0.0127 |

## Hibatérkép: amerikai és európai teszt-RMSE aránya

| Pár | Medián (400 cella) | A 10% legnagyobb prémiumú cella | Határ-közeli cellák | Többi cella | ITM × hosszú T (sorok) | Lehívási sorok | Spearman ρ (arány és prémium) | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_retune_p30` | 0.918 | 0.879 | 0.922 | 0.916 | 0.933 | 0.845 | 0.0507 | 0.312 |
| `resnet_retune_p30` | 0.837 | 0.696 | 0.843 | 0.831 | 0.840 | 0.913 | −0.0917 | 0.0671 |

## Arbitrázsmentesség: technikai ellenőrzések

| Ellenőrzés | Eredmény |
|---|---:|
| A háló float32 bemenete bitre egyezik a parquettel (mindhárom split) | igen |
| float64-ben a legnagyobb abszolút eltérés | 2.11e-15 |
| Autograd és véges differencia: a delta relatív eltérésének legnagyobb mediánja (42 modell) | 4.22e-8 |
| Autograd és véges differencia: a théta relatív eltérésének legnagyobb mediánja (42 modell) | 2.19e-9 |
| 1e-3 feletti eltérésű sorok legnagyobb aránya, delta | 6.95% |
| 1e-3 feletti eltérésű sorok legnagyobb aránya, théta | 4.05% |
| BS put mint modell: 2., 3., 4. és 6. pont, τ = 1e-6, megsértés nélkül | igen |
| CRR-500 címke a rácsszeleten (r = 0.05, σ = 0.2625): 1., 2'., 3., 4., 5. pont megsértés nélkül | igen |
| CRR-500 címke a rácsszeleten: a 2. pont megsértései | 3 |
| Parquet-címkék a teszthalmazon (150 000 sor): a 2. pont megsértései, τ = 0 | 7 532 |
| Parquet-címkék a teszthalmazon (150 000 sor): a 2. pont megsértései, τ = 1e-6 | 3 068 |

## Megsértési arány, amerikai put modellek, teszthalmaz (τ = 1e-6 / τ = teszt-RMSE)

| Modell | 1. P ≥ max(K−S,0) | 2. P_am ≥ P_BS | 3. P ≤ K | 4a. monoton csökken S-ben | 4b. ∂P/∂S ≥ −1 | 5. monoton nő T-ben | 6. konvex S-ben |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer, uniform p30 | 9.13% / 2.97% | 25.12% / 3.93% | 0.00% / 0.00% | 0.00% / 0.00% | 0.04% / 0.00% | 5.67% / 0.00% | 12.58% / 0.00% |
| DeepMLPPricer, uniform p30 | 2.63% / 0.44% | 18.01% / 3.77% | 0.00% / 0.00% | 0.00% / 0.00% | 7.65% / 0.00% | 6.57% / 0.00% | 36.65% / 0.00% |
| DenseMLPPricer, uniform p30 | 18.88% / 9.63% | 38.85% / 8.03% | 0.00% / 0.00% | 0.00% / 0.00% | 4.03% / 0.00% | 2.52% / 0.00% | 4.72% / 0.00% |
| ResNetPricer, uniform p30 | 11.03% / 4.05% | 23.83% / 6.62% | 0.00% / 0.00% | 3.22% / 0.00% | 4.49% / 0.00% | 10.86% / 0.00% | 35.41% / 0.00% |
| GELUResNetPricer, uniform p30 | 15.92% / 8.17% | 31.89% / 5.46% | 0.00% / 0.00% | 4.46% / 0.00% | 6.23% / 0.00% | 7.90% / 0.00% | 0.00% / 0.00% |
| HighwayPricer, uniform p30 | 17.38% / 8.34% | 41.24% / 6.81% | 0.00% / 0.00% | 3.22% / 0.00% | 0.86% / 0.00% | 7.40% / 0.00% | 0.71% / 0.00% |
| FINNPricer, uniform p30 | 15.25% / 7.44% | 30.16% / 4.75% | 0.00% / 0.00% | 4.31% / 0.00% | 6.50% / 0.00% | 8.34% / 0.00% | 0.01% / 0.00% |
| MLPPricer, transfer p30 | 9.43% / 1.47% | 14.09% / 2.30% | 0.00% / 0.00% | 1.09% / 0.00% | 4.25% / 0.00% | 5.64% / 0.00% | 20.14% / 0.00% |
| DeepMLPPricer, transfer p30 | 10.14% / 2.18% | 27.54% / 6.51% | 0.00% / 0.00% | 1.29% / 0.00% | 5.89% / 0.01% | 6.05% / 0.00% | 34.15% / 0.20% |
| DenseMLPPricer, transfer p30 | 13.70% / 5.90% | 21.25% / 4.87% | 0.00% / 0.00% | 2.67% / 0.00% | 4.00% / 0.00% | 7.32% / 0.00% | 6.24% / 0.00% |
| ResNetPricer, transfer p30 | 9.44% / 1.47% | 22.37% / 4.86% | 0.00% / 0.00% | 0.61% / 0.00% | 3.14% / 0.00% | 6.74% / 0.00% | 34.73% / 0.00% |
| GELUResNetPricer, transfer p30 | 16.35% / 9.60% | 29.80% / 6.62% | 0.00% / 0.00% | 0.64% / 0.00% | 2.98% / 0.00% | 5.31% / 0.00% | 10.58% / 0.00% |
| HighwayPricer, transfer p30 | 12.34% / 5.30% | 22.08% / 5.89% | 0.00% / 0.00% | 3.55% / 0.00% | 5.97% / 0.00% | 8.37% / 0.00% | 3.72% / 0.00% |
| FINNPricer, transfer p30 | 12.93% / 5.80% | 20.47% / 5.51% | 0.00% / 0.00% | 2.99% / 0.00% | 5.18% / 0.00% | 8.69% / 0.00% | 8.59% / 0.00% |
| MLPPricer, retune p30 | 8.61% / 1.17% | 12.90% / 1.93% | 0.00% / 0.00% | 0.09% / 0.00% | 4.58% / 0.00% | 5.45% / 0.00% | 20.05% / 0.00% |
| DeepMLPPricer, retune p30 | 7.18% / 1.10% | 25.70% / 4.41% | 0.00% / 0.00% | 1.28% / 0.00% | 6.44% / 0.00% | 8.37% / 0.00% | 35.31% / 0.00% |
| DenseMLPPricer, retune p30 | 16.15% / 9.02% | 29.08% / 7.23% | 0.00% / 0.00% | 0.88% / 0.00% | 2.40% / 0.00% | 3.07% / 0.00% | 9.25% / 0.00% |
| ResNetPricer, retune p30 | 8.99% / 3.12% | 21.66% / 4.54% | 0.00% / 0.00% | 0.70% / 0.00% | 2.67% / 0.00% | 7.37% / 0.00% | 28.66% / 0.00% |
| GELUResNetPricer, retune p30 | 15.13% / 7.58% | 28.49% / 8.79% | 0.00% / 0.00% | 0.98% / 0.00% | 3.60% / 0.00% | 5.04% / 0.00% | 9.49% / 0.00% |
| HighwayPricer, retune p30 | 16.90% / 10.45% | 24.66% / 5.74% | 0.00% / 0.00% | 2.18% / 0.00% | 2.36% / 0.00% | 6.53% / 0.00% | 5.82% / 0.00% |
| FINNPricer, retune p30 | 11.92% / 4.77% | 14.84% / 3.77% | 0.00% / 0.00% | 1.29% / 0.00% | 4.46% / 0.01% | 8.35% / 0.00% | 10.28% / 0.00% |

## Megsértési arány, amerikai put modellek, rács (τ = 1e-6 / τ = teszt-RMSE)

| Modell | 1. P ≥ max(K−S,0) | 2. P_am ≥ P_BS | 3. P ≤ K | 4a. monoton csökken S-ben | 4b. ∂P/∂S ≥ −1 | 5. monoton nő T-ben | 6. konvex S-ben |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer, uniform p30 | 15.45% / 5.93% | 30.52% / 7.43% | 0.00% / 0.00% | 0.00% / 0.00% | 0.11% / 0.00% | 8.67% / 0.00% | 13.03% / 0.00% |
| DeepMLPPricer, uniform p30 | 6.88% / 1.58% | 21.70% / 4.31% | 0.00% / 0.00% | 0.03% / 0.00% | 8.11% / 0.00% | 8.74% / 0.00% | 36.96% / 0.00% |
| DenseMLPPricer, uniform p30 | 23.48% / 14.34% | 44.57% / 13.77% | 0.00% / 0.00% | 0.00% / 0.00% | 5.47% / 0.00% | 4.76% / 0.00% | 6.38% / 0.00% |
| ResNetPricer, uniform p30 | 19.60% / 10.83% | 36.08% / 17.15% | 0.00% / 0.00% | 5.45% / 0.00% | 5.36% / 0.00% | 12.78% / 0.03% | 35.84% / 0.00% |
| GELUResNetPricer, uniform p30 | 20.77% / 13.98% | 34.89% / 11.30% | 0.00% / 0.00% | 4.85% / 0.00% | 6.97% / 0.00% | 13.17% / 0.00% | 0.00% / 0.00% |
| HighwayPricer, uniform p30 | 21.37% / 12.60% | 43.08% / 11.16% | 0.00% / 0.00% | 3.91% / 0.00% | 1.78% / 0.00% | 15.56% / 0.00% | 1.15% / 0.00% |
| FINNPricer, uniform p30 | 19.56% / 12.98% | 33.88% / 10.44% | 0.00% / 0.00% | 4.79% / 0.00% | 6.93% / 0.00% | 14.65% / 0.00% | 0.04% / 0.00% |
| MLPPricer, transfer p30 | 15.54% / 3.85% | 27.16% / 8.76% | 0.00% / 0.00% | 2.45% / 0.00% | 5.22% / 0.00% | 11.25% / 0.06% | 21.07% / 0.00% |
| DeepMLPPricer, transfer p30 | 16.64% / 4.69% | 34.76% / 9.51% | 0.00% / 0.00% | 2.17% / 0.00% | 6.46% / 0.02% | 8.40% / 0.00% | 35.19% / 0.27% |
| DenseMLPPricer, transfer p30 | 17.14% / 11.65% | 29.27% / 13.02% | 0.00% / 0.00% | 6.84% / 0.00% | 6.49% / 0.00% | 13.31% / 0.00% | 10.14% / 0.00% |
| ResNetPricer, transfer p30 | 17.01% / 4.37% | 30.42% / 10.09% | 0.00% / 0.00% | 1.36% / 0.00% | 4.91% / 0.00% | 11.59% / 0.00% | 35.07% / 0.00% |
| GELUResNetPricer, transfer p30 | 23.50% / 18.83% | 39.16% / 18.21% | 0.00% / 0.00% | 1.55% / 0.00% | 4.04% / 0.00% | 13.10% / 0.00% | 14.23% / 0.00% |
| HighwayPricer, transfer p30 | 17.34% / 11.98% | 32.07% / 13.91% | 0.00% / 0.00% | 6.36% / 0.00% | 8.32% / 0.00% | 16.27% / 0.00% | 7.03% / 0.00% |
| FINNPricer, transfer p30 | 17.54% / 11.70% | 32.50% / 14.67% | 0.00% / 0.00% | 7.13% / 0.00% | 6.07% / 0.00% | 15.92% / 0.00% | 10.46% / 0.00% |
| MLPPricer, retune p30 | 13.22% / 2.58% | 23.48% / 6.39% | 0.00% / 0.00% | 0.14% / 0.00% | 5.26% / 0.00% | 10.28% / 0.03% | 20.08% / 0.00% |
| DeepMLPPricer, retune p30 | 13.89% / 4.30% | 30.97% / 7.81% | 0.00% / 0.00% | 3.10% / 0.00% | 7.32% / 0.01% | 13.12% / 0.00% | 37.15% / 0.00% |
| DenseMLPPricer, retune p30 | 22.68% / 17.02% | 39.13% / 17.35% | 0.00% / 0.00% | 3.10% / 0.00% | 4.60% / 0.00% | 8.57% / 0.00% | 12.53% / 0.00% |
| ResNetPricer, retune p30 | 15.69% / 6.35% | 29.79% / 9.58% | 0.00% / 0.00% | 2.30% / 0.00% | 4.49% / 0.00% | 12.55% / 0.00% | 29.13% / 0.00% |
| GELUResNetPricer, retune p30 | 21.89% / 16.12% | 36.08% / 17.45% | 0.00% / 0.00% | 2.72% / 0.00% | 4.85% / 0.00% | 13.03% / 0.00% | 14.38% / 0.00% |
| HighwayPricer, retune p30 | 22.48% / 16.32% | 35.07% / 14.22% | 0.00% / 0.00% | 6.28% / 0.00% | 4.65% / 0.00% | 14.93% / 0.00% | 8.92% / 0.00% |
| FINNPricer, retune p30 | 17.21% / 11.22% | 22.84% / 10.67% | 0.00% / 0.00% | 3.32% / 0.00% | 5.43% / 0.03% | 16.49% / 0.22% | 15.78% / 0.00% |

## Megsértési arány szintenként, amerikai put (medián a 7 architektúrán, teszthalmaz)

| Szint | τ | 1. P ≥ max(K−S,0) | 2. P_am ≥ P_BS | 3. P ≤ K | 4a. monoton csökken S-ben | 4b. ∂P/∂S ≥ −1 | 5. monoton nő T-ben | 6. konvex S-ben |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform p30 | 0 | 15.26% | 30.17% | 0.00% | 3.22% | 4.50% | 7.47% | 6.73% |
| transfer p30 | 0 | 12.34% | 22.09% | 0.00% | 1.31% | 4.29% | 6.82% | 10.87% |
| retune p30 | 0 | 11.92% | 24.68% | 0.00% | 1.00% | 3.62% | 6.75% | 10.54% |
| uniform p30 | 1e-6 | 15.25% | 30.16% | 0.00% | 3.22% | 4.49% | 7.40% | 4.72% |
| transfer p30 | 1e-6 | 12.34% | 22.08% | 0.00% | 1.29% | 4.25% | 6.74% | 10.58% |
| retune p30 | 1e-6 | 11.92% | 24.66% | 0.00% | 0.98% | 3.60% | 6.53% | 10.28% |
| uniform p30 | RMSE | 7.44% | 5.46% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| transfer p30 | RMSE | 5.30% | 5.51% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| retune p30 | RMSE | 4.77% | 4.54% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Megsértési arány szintenként, európai put (medián a 7 architektúrán, teszthalmaz, τ = 1e-6 / τ = teszt-RMSE)

| Szint | 1e. P ≥ max(Ke^{−rT}−S,0) | 3. P ≤ K | 4a. monoton csökken S-ben | 4b. ∂P/∂S ≥ −1 | 6. konvex S-ben |
|---|---:|---:|---:|---:|---:|
| uniform p30 | 12.76% / 6.65% | 0.00% / 0.00% | 0.46% / 0.00% | 3.67% / 0.00% | 4.64% / 0.00% |
| transfer p30 | 8.35% / 3.27% | 0.00% / 0.00% | 1.80% / 0.00% | 3.00% / 0.00% | 8.58% / 0.00% |
| retune p30 | 9.20% / 3.92% | 0.00% / 0.00% | 1.05% / 0.00% | 2.28% / 0.00% | 8.73% / 0.00% |

## Megsértési arány régiónként, MLPPricer retune p30 (teszthalmaz, τ = 1e-6)

| Régió | 1. P ≥ max(K−S,0) | 2. P_am ≥ P_BS | 3. P ≤ K | 4a. monoton csökken S-ben | 4b. ∂P/∂S ≥ −1 | 5. monoton nő T-ben | 6. konvex S-ben |
|---|---:|---:|---:|---:|---:|---:|---:|
| ITM (S/K<0,9) × rövid (T<0,25) | 35.55% | 16.77% | 0.00% | 0.00% | 23.78% | 17.25% | 31.87% |
| ITM (S/K<0,9) × közép (0,25≤T<1) | 15.22% | 3.39% | 0.00% | 0.00% | 13.97% | 13.98% | 26.76% |
| ITM (S/K<0,9) × hosszú (1≤T≤2) | 9.12% | 2.51% | 0.00% | 0.00% | 6.47% | 10.59% | 25.30% |
| ATM (0,9≤S/K<1,1) × rövid (T<0,25) | 2.35% | 44.28% | 0.00% | 0.00% | 0.38% | 0.16% | 8.93% |
| ATM (0,9≤S/K<1,1) × közép (0,25≤T<1) | 0.50% | 12.34% | 0.00% | 0.00% | 0.05% | 0.88% | 18.74% |
| ATM (0,9≤S/K<1,1) × hosszú (1≤T≤2) | 0.85% | 2.85% | 0.00% | 0.00% | 0.09% | 0.69% | 20.78% |
| OTM (S/K≥1,1) × rövid (T<0,25) | 18.98% | 48.82% | 0.00% | 1.79% | 0.00% | 0.00% | 7.07% |
| OTM (S/K≥1,1) × közép (0,25≤T<1) | 6.23% | 27.53% | 0.00% | 0.00% | 0.00% | 0.00% | 12.05% |
| OTM (S/K≥1,1) × hosszú (1≤T≤2) | 3.41% | 11.13% | 0.00% | 0.00% | 0.00% | 1.16% | 17.54% |

## A leggyakoribb megsértés modellenként (teszthalmaz, τ = 1e-6)

| Cél | Leggyakoribb megsértés | Modellek | A legtöbb megsértés régiója (módusz) |
|---|---|---:|---|
| amerikai put | 2. P_am ≥ P_BS | 13 | OTM (S/K≥1,1) × közép (0,25≤T<1) |
| amerikai put | 6. konvex S-ben | 8 | ITM (S/K<0,9) × hosszú (1≤T≤2) |
| európai put | 1e. P ≥ max(Ke^{−rT}−S,0) | 11 | ITM (S/K<0,9) × közép (0,25≤T<1) |
| európai put | 6. konvex S-ben | 10 | ITM (S/K<0,9) × hosszú (1≤T≤2) |

## Sebesség: mérési környezet

| Jellemző | Érték |
|---|---|
| CPU | AMD Ryzen 7 7735HS with Radeon Graphics (8 mag, 16 szál) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Szoftver | torch 2.6.0+cu124, CUDA 12.4, Python 3.13.7, NumPy 2.3.3 |
| torch-szálszám | 8 |
| CPU-terhelés előtte / utána (%) | [0.0, 3.0, 4.0] / [4.0, 4.0, 1.0] |
| Szabad RAM előtte / utána (MB) | 5 949 / 5 638 |
| Energiaséma | Performance |
| Akkumulátor | töltöttség 80%, BatteryStatus = 2 |

## Sebesség és pontosság (a CRR soronként hívott, egyszálú Python-implementáció)

| Módszer | Beállítás | Idő áranként (s, medián) | Gyorsulás a CRR-hez | RMSE ($) | A pontosság forrása |
|---|---|---:|---:|---:|---|
| NN, MLPPricer | GPU, batch 1e6 | 2.93e-8 | 4.22e5 | 0.0755 | levels_table.csv (retune p30, teszthalmaz) |
| NN, DenseMLPPricer | GPU, batch 1e6 | 1.07e-7 | 1.16e5 | 0.387 | levels_table.csv (retune p30, teszthalmaz) |
| NN, DeepMLPPricer | GPU, batch 1e6 | 1.52e-7 | 8.13e4 | 0.310 | levels_table.csv (retune p30, teszthalmaz) |
| BS put (európai) | CPU, batch 1e6 | 1.74e-7 | 7.13e4 | 1.53 | BS put vs. amerikai címke, teljes teszthalmaz (≈ a prémium) |
| NN, GELUResNetPricer | GPU, batch 1e6 | 2.18e-7 | 5.69e4 | 0.187 | levels_table.csv (retune p30, teszthalmaz) |
| NN, FINNPricer | GPU, batch 1e6 | 2.28e-7 | 5.44e4 | 0.127 | levels_table.csv (retune p30, teszthalmaz) |
| NN, ResNetPricer | GPU, batch 1e6 | 2.35e-7 | 5.28e4 | 0.243 | levels_table.csv (retune p30, teszthalmaz) |
| NN, HighwayPricer | GPU, batch 1e6 | 4.02e-7 | 3.08e4 | 0.287 | levels_table.csv (retune p30, teszthalmaz) |
| NN, MLPPricer | CPU, batch 1 | 1.16e-4 | 107 | 0.0755 | levels_table.csv (retune p30, teszthalmaz) |
| NN, DeepMLPPricer | CPU, batch 1 | 3.10e-4 | 39.9 | 0.310 | levels_table.csv (retune p30, teszthalmaz) |
| NN, DenseMLPPricer | CPU, batch 1 | 3.41e-4 | 36.3 | 0.387 | levels_table.csv (retune p30, teszthalmaz) |
| NN, MLPPricer | GPU, batch 1 | 4.90e-4 | 25.3 | 0.0755 | levels_table.csv (retune p30, teszthalmaz) |
| NN, GELUResNetPricer | CPU, batch 1 | 5.32e-4 | 23.3 | 0.187 | levels_table.csv (retune p30, teszthalmaz) |
| NN, HighwayPricer | CPU, batch 1 | 6.54e-4 | 19.0 | 0.287 | levels_table.csv (retune p30, teszthalmaz) |
| NN, FINNPricer | CPU, batch 1 | 6.55e-4 | 18.9 | 0.127 | levels_table.csv (retune p30, teszthalmaz) |
| NN, DenseMLPPricer | GPU, batch 1 | 6.58e-4 | 18.8 | 0.387 | levels_table.csv (retune p30, teszthalmaz) |
| NN, ResNetPricer | CPU, batch 1 | 6.78e-4 | 18.3 | 0.243 | levels_table.csv (retune p30, teszthalmaz) |
| NN, GELUResNetPricer | GPU, batch 1 | 1.19e-3 | 10.5 | 0.187 | levels_table.csv (retune p30, teszthalmaz) |
| NN, FINNPricer | GPU, batch 1 | 1.28e-3 | 9.71 | 0.127 | levels_table.csv (retune p30, teszthalmaz) |
| NN, DeepMLPPricer | GPU, batch 1 | 1.35e-3 | 9.17 | 0.310 | levels_table.csv (retune p30, teszthalmaz) |
| NN, HighwayPricer | GPU, batch 1 | 1.46e-3 | 8.50 | 0.287 | levels_table.csv (retune p30, teszthalmaz) |
| NN, ResNetPricer | GPU, batch 1 | 1.69e-3 | 7.33 | 0.243 | levels_table.csv (retune p30, teszthalmaz) |
| CRR-500 | CPU, soronként | 0.0124 | 1.00 | 4.73e-3 | CRR-500 vs. CRR-1000, 200 sor (a címke saját hibája) |
| LSM | CPU, soronként | 0.555 | 0.0223 | 0.0450 | LSM-QA vs. CRR-címke, 275 tesztsor |

## Összevetés a tanítóadat generálásakor mért idővel

| Módszer | Most | A generáláskor | Arány |
|---|---:|---:|---:|
| CRR-amerikai, ms/ár | 12.4 (a futás végén 12.4) | 8.29 | 1.49 |
| CRR-európai batch, ms/ár (most 1000, a generáláskor 1M sor) | 0.916 | 1.07 | 0.853 |
| LSM, s/ár (20 sor, zárójelben a QA lépésszámára skálázva) | 0.550 (0.532) | 0.369 | 1.44 |

## NN-inferencia: idő áranként (s, medián)

| Architektúra | GPU, batch 1 | GPU, batch 1e3 | GPU, batch 1e5 | GPU, batch 1e6 | CPU, batch 1 | CPU, batch 1e3 | CPU, batch 1e5 | CPU, batch 1e6 | CPU, 1 szál, batch 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLPPricer | 4.90e-4 | 4.95e-7 | 3.28e-8 | 2.93e-8 | 1.16e-4 | 6.55e-7 | 6.16e-7 | 8.17e-7 | 1.48e-4 |
| DeepMLPPricer | 1.35e-3 | 8.13e-7 | 1.52e-7 | 1.52e-7 | 3.10e-4 | 2.26e-6 | 3.06e-6 | 3.08e-6 | 3.07e-4 |
| DenseMLPPricer | 6.58e-4 | 1.14e-6 | 1.06e-7 | 1.07e-7 | 3.41e-4 | 1.66e-6 | 2.05e-6 | 1.90e-6 | 2.58e-4 |
| ResNetPricer | 1.69e-3 | 1.62e-6 | 2.32e-7 | 2.35e-7 | 6.78e-4 | 3.49e-6 | 4.73e-6 | 4.73e-6 | 6.95e-4 |
| GELUResNetPricer | 1.19e-3 | 1.15e-6 | 2.14e-7 | 2.18e-7 | 5.32e-4 | 3.55e-6 | 4.50e-6 | 4.40e-6 | 4.99e-4 |
| HighwayPricer | 1.46e-3 | 1.29e-6 | 3.98e-7 | 4.02e-7 | 6.54e-4 | 4.79e-6 | 6.73e-6 | 7.38e-6 | 6.15e-4 |
| FINNPricer | 1.28e-3 | 1.38e-6 | 2.23e-7 | 2.28e-7 | 6.55e-4 | 4.35e-6 | 4.55e-6 | 4.55e-6 | 6.20e-4 |
