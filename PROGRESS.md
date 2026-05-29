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

## Passo 8 - Estrazione feature EDA wrist

- Script creati:
  - `src/extract_eda_features_all.py`
  - `src/validate_eda_csv.py`
- Segnale usato: `signal["wrist"]["EDA"]`, campionato a `4 Hz`.
- Segmentazione coerente con BVP: finestre di `60 s`, passo di `30 s`, purezza label minima `0.70`.
- Feature estratte: statistiche descrittive EDA, slope e derivate, componente tonica, componente fasica positiva e risposte SCR rilevate tramite picchi.
- File generati:
  - `data_features/eda_features_all_60s.csv`
  - `data_features/eda_features_train_s2_s16_60s.csv`
  - `data_features/eda_features_test_s17_60s.csv`
- Dimensioni dataset: `1069` finestre, `36` colonne, `15` soggetti; train `995` righe e test S17 `74` righe.
- Distribuzione label binarie: non-stress `748`, stress `321`.
- Controlli superati: nessun NaN/infinito nelle feature, schema e label validi, finestre EDA allineate riga per riga al CSV BVP.


## Passo 9 - EDA wrist + kNN LOSO

- Script creato: `src/train_eda_knn_loso.py`
- Dataset usato: `data_features/eda_features_all_60s.csv`
- Feature usate: `eda_mean, eda_std, eda_min, eda_max, eda_range, eda_median, eda_iqr, eda_rms, eda_energy, eda_skew, eda_kurtosis, eda_slope, eda_derivative_mean, eda_derivative_std, eda_derivative_max, eda_tonic_mean, eda_tonic_std, eda_tonic_slope, eda_phasic_mean, eda_phasic_std, eda_phasic_max, eda_phasic_auc, eda_scr_count, eda_scr_rate_per_min, eda_scr_amplitude_mean, eda_scr_amplitude_std, eda_scr_amplitude_max, eda_scr_prominence_mean`
- Validazione: Leave-One-Subject-Out per soggetto, con `StandardScaler` fittato solo sul training fold.
- Modello baseline: `KNeighborsClassifier(n_neighbors=9, weights="distance")`
- Risultati principali:
  - Accuracy media fold: 0.8561 (85.61%)
  - F1 media fold: 0.7354 (73.54%)
  - Accuracy aggregata globale: 0.8559 (85.59%)
  - F1 aggregata globale: 0.7548 (75.48%)
  - Confusion matrix aggregata: TN=678, FP=70, FN=84, TP=237
- File generati:
  - `results/eda_knn_loso_per_subject.csv`
  - `results/eda_knn_summary.txt`
- Osservazioni:
  - Il modello predice entrambe le classi.
  - Soggetti problematici secondo soglia F1: S14, S17, S6
  - Questa esecuzione e' la baseline EDA; non include ancora tuning degli iperparametri.

## Report HTML EDA

- File creato: `results/report_eda_knn_loso.html`
- Contenuto: pipeline EDA completa, segmentazione, feature statistiche/toniche/fasiche/SCR, validazione LOSO, metriche globali, risultati per soggetto, feature piu' importanti e spiegazione pronta per l'orale.

## Demo predizioni BVP ed EDA

- Script disponibili:
  - `src/demo_predict_bvp.py`
  - `src/demo_predict_eda.py`
- Uso: ciascuna demo allena il rispettivo kNN su S2-S16 e mostra `15` predizioni bilanciate estratte dal test set S17.
- Output mostrato per finestra: label vera, label predetta (`stress`/`non-stress`), probabilita' stimata di stress ed esito `GIUSTO`/`SBAGLIATO`.
- Comandi pronti in `riga_da_eseguire.txt`.
- CSV generati dalle demo:
  - `results/demo_predictions_bvp_s17.csv`
  - `results/demo_predictions_eda_s17.csv`
