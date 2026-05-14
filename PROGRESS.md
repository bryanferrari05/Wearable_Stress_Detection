# PROGRESS

## Passo 7 - BVP wrist + kNN LOSO

- Script creato: `src/train_bvp_knn_loso.py`
- Dataset usato: `data_features/bvp_features_all_60s.csv`
- CSV rigenerato da `WESAD/` con feature statistiche, peak detection e HR/HRV semplici.
- Feature usate: `bvp_mean, bvp_std, bvp_min, bvp_max, bvp_range, bvp_median, bvp_iqr, bvp_rms, bvp_energy, bvp_skew, bvp_kurtosis, bvp_peak_count, bvp_valid_ibi_count, bvp_peak_prominence_mean, bvp_peak_prominence_std, bvp_ibi_mean, bvp_ibi_std, bvp_ibi_min, bvp_ibi_max, bvp_hr_mean, bvp_hr_std, bvp_hr_min, bvp_hr_max, bvp_rmssd, bvp_sdnn`
- Feature avanzate mancanti nel CSV: `nessuna`
- Feature fisiologiche HR/HRV mancanti nel CSV: `nessuna`
- Validazione: Leave-One-Subject-Out per soggetto, con `StandardScaler` fittato solo sul training fold.
- Modello: `KNeighborsClassifier(n_neighbors=9, weights="distance")`
- Risultati principali:
  - Accuracy media fold: 0.8277 (82.77%)
  - F1 media fold: 0.6990 (69.90%)
  - Accuracy aggregata globale: 0.8279 (82.79%)
  - F1 aggregata globale: 0.7089 (70.89%)
  - Confusion matrix aggregata: TN=661, FP=87, FN=97, TP=224
- File generati:
  - `results/bvp_knn_loso_per_subject.csv`
  - `results/bvp_knn_summary.txt`
- Osservazioni:
  - Il modello predice entrambe le classi.
  - Soggetti problematici secondo soglia F1: S15, S9
  - Confronto paper indicativo: BVP + kNN circa 82.06% accuracy e 78.94% F1.

## Tuning leggero kNN BVP

- Script creato: `src/tune_bvp_knn_loso.py`
- Output:
  - `results/bvp_knn_tuning_results.csv`
  - `results/bvp_knn_tuning_summary.txt`
- Esperimenti: `k=[1, 3, 5, 7, 9, 11, 15, 21]` con `weights=['uniform', 'distance']`
- Nota: tuning eseguito prima dell'aggiunta delle feature peak detection/HRV.
- Validazione: Leave-One-Subject-Out, con `StandardScaler` fittato solo sul training fold.
- Migliore configurazione per F1 aggregato: `k=21`, `weights="distance"`
- Risultati migliori:
  - Accuracy aggregata: 0.7774 (77.74%)
  - Precision aggregata: 0.6751 (67.51%)
  - Recall aggregata: 0.4984 (49.84%)
  - F1 aggregato: 0.5735 (57.35%)
  - Confusion matrix: TN=671, FP=77, FN=161, TP=160

## Report HTML

- File creato: `results/report_bvp_knn_loso.html`
- Contenuto: timeline del lavoro, test effettuati, metriche per ogni passo, spiegazione delle feature statistiche, peak detection, HR/HRV semplici e motivazione feature-per-feature.
