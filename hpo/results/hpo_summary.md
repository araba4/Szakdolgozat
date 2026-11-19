## Teszt-RMSE és validációs MSE

| Architektúra | Paraméter | Teszt-RMSE, kiindulási | Teszt-RMSE, HPO | Változás | Val MSE, kiindulási | Val MSE, HPO |
|---|---:|---:|---:|---:|---:|---:|
| MLPPricer | 30 901 | 4.60e-3 | 9.78e-4 | −78.7% | 2.06e-5 | 9.01e-7 |
| DeepMLPPricer | 267 265 | 6.06e-3 | 2.47e-3 | −59.2% | 3.63e-5 | 6.05e-6 |
| DenseMLPPricer | 101 381 | 1.80e-2 | 2.04e-3 | −88.7% | 3.23e-4 | 4.06e-6 |
| ResNetPricer | 399 873 | 6.22e-3 | 4.28e-3 | −31.1% | 3.85e-5 | 1.81e-5 |
| GELUResNetPricer | 398 337 | 9.29e-3 | 3.88e-3 | −58.3% | 8.50e-5 | 1.48e-5 |
| HighwayPricer | 527 873 | 1.49e-2 | 2.95e-3 | −80.2% | 2.20e-4 | 8.55e-6 |
| FINNPricer | 402 882 | 9.14e-3 | 2.98e-3 | −67.4% | 8.24e-5 | 8.67e-6 |

## Legjobb hiperparaméterek

| Architektúra | Trial | lr | weight_decay | dropout |
|---|---:|---:|---:|---:|
| MLPPricer | 25 | 1.39e-3 | 1.36e-6 | – |
| DeepMLPPricer | 30 | 1.50e-3 | 1.38e-6 | 0.018 |
| DenseMLPPricer | 30 | 3.05e-3 | 1.08e-6 | 0.012 |
| ResNetPricer | 40 | 4.50e-3 | 2.90e-4 | 0.325 |
| GELUResNetPricer | 40 | 2.39e-4 | 1.10e-6 | 0.122 |
| HighwayPricer | 40 | 2.21e-3 | 1.02e-6 | 0.406 |
| FINNPricer | 40 | 3.37e-3 | 1.07e-6 | 0.027 |
