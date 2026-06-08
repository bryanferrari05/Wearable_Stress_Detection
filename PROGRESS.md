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


## Test EDA feature ridotte

- Script creato: `src/train_eda_knn_loso_reduced.py`
- Dataset usato: `data_features/eda_features_all_60s.csv`
- Feature complete: `28`
- Feature ridotte usate: `16`
- Feature rimosse per ridondanza: `eda_min, eda_max, eda_median, eda_iqr, eda_rms, eda_energy, eda_tonic_mean, eda_tonic_slope, eda_phasic_mean, eda_phasic_max, eda_scr_rate_per_min, eda_scr_amplitude_mean`
- Validazione: Leave-One-Subject-Out per soggetto, con `StandardScaler` fittato solo sul training fold.
- Modello: `KNeighborsClassifier(n_neighbors=9, weights="distance")`
- Risultati feature ridotte:
  - Accuracy aggregata: 0.8597 (85.97%)
  - Precision aggregata: 0.7785 (77.85%)
  - Recall aggregata: 0.7445 (74.45%)
  - F1 aggregato: 0.7611 (76.11%)
  - Confusion matrix aggregata: TN=680, FP=68, FN=82, TP=239
- Confronto con baseline EDA completa:
  - Delta accuracy: +0.0038 (0.38%)
  - Delta precision: +0.0065 (0.65%)
  - Delta recall: +0.0062 (0.62%)
  - Delta F1: +0.0063 (0.63%)
- File generati:
  - `results/eda_knn_loso_reduced_per_subject.csv`
  - `results/eda_knn_reduced_summary.txt`
- Soggetti problematici secondo soglia F1: S14, S17


## Test EDA normalizzazione baseline soggetto

- Script creato: `src/train_eda_knn_loso_baseline_norm.py`
- Dataset usato: `data_features/eda_features_all_60s.csv`
- Assunzione: baseline personale disponibile per ogni soggetto, usando solo finestre con `original_label=1`.
- Feature set testati: `reduced_absolute_16`, `baseline_delta_10`, `baseline_zscore_10`, `reduced_plus_delta_26`, `reduced_plus_zscore_26`, `reduced_plus_delta_zscore_36`
- Migliore feature set: `reduced_plus_delta_zscore_36` con `36` feature.
- Risultati migliori:
  - Accuracy aggregata: 0.8952 (89.52%)
  - Precision aggregata: 0.8828 (88.28%)
  - Recall aggregata: 0.7508 (75.08%)
  - F1 aggregato: 0.8114 (81.14%)
  - Confusion matrix: TN=716, FP=32, FN=80, TP=241
- Delta F1 vs EDA ridotta 16 feature precedente: +0.0503 (5.03%)
- File generati:
  - `results/eda_knn_baseline_norm_results.csv`
  - `results/eda_knn_baseline_norm_best_per_subject.csv`
  - `results/eda_knn_baseline_norm_summary.txt`

## Passo 10 - Estrazione feature ECG chest

- Script creati:
  - `src/extract_ecg_chest_features_all.py`
  - `src/validate_ecg_chest_csv.py`
- Segnale usato: `signal["chest"]["ECG"]`, campionato a `700 Hz`.
- Segmentazione coerente con BVP/EDA: finestre di `60 s`, passo di `30 s`, purezza label minima `0.70`.
- Feature estratte: statistiche sul segnale ECG filtrato bandpass, derivate, R-peak, RR interval, heart rate e HRV time-domain semplice.
- File CSV attesi dopo l'estrazione:
  - `data_features/ecg_chest_features_all_60s.csv`
  - `data_features/ecg_chest_features_train_s2_s16_60s.csv`
  - `data_features/ecg_chest_features_test_s17_60s.csv`
- Validazione prevista: schema colonne, label, purezza, NaN/infinito, coerenza range/rate e allineamento finestre con il CSV BVP se presente.
- Nota esecuzione locale: gli script compilano correttamente, ma in questo workspace visibile non sono presenti `data_raw/`, `WESAD/` o cartelle soggetto WESAD da cui generare subito i CSV.

## Passo 11 - ECG chest + Random Forest LOSO

- Script creato: `src/train_ecg_chest_rf_loso.py`
- Dataset usato: `data_features/ecg_chest_features_all_60s.csv`
- Feature usate: `ecg_mean, ecg_std, ecg_min, ecg_max, ecg_range, ecg_median, ecg_iqr, ecg_rms, ecg_energy, ecg_skew, ecg_kurtosis, ecg_derivative_mean, ecg_derivative_std, ecg_derivative_max_abs, ecg_r_peak_count, ecg_valid_rr_count, ecg_r_peak_rate_per_min, ecg_r_peak_prominence_mean, ecg_r_peak_prominence_std, ecg_rr_mean, ecg_rr_std, ecg_rr_min, ecg_rr_max, ecg_hr_mean, ecg_hr_std, ecg_hr_min, ecg_hr_max, ecg_hr_range, ecg_rmssd, ecg_sdnn, ecg_nn50_count, ecg_pnn50, ecg_cvnn`
- Validazione: Leave-One-Subject-Out per soggetto.
- Modello baseline: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42)`
- Scaling: non usato, perche' Random Forest non richiede `StandardScaler`.
- Risultati principali:
  - Accuracy media fold: 0.7493 (74.93%)
  - F1 media fold: 0.5480 (54.80%)
  - Accuracy aggregata globale: 0.7484 (74.84%)
  - F1 aggregata globale: 0.5943 (59.43%)
  - Confusion matrix aggregata: TN=603, FP=145, FN=124, TP=197
- File generati:
  - `results/ecg_chest_rf_loso_per_subject.csv`
  - `results/ecg_chest_rf_feature_importance.csv`
  - `results/ecg_chest_rf_summary.txt`
- Osservazioni:
  - Il modello predice entrambe le classi.
  - Soggetti problematici secondo soglia F1: S15, S2, S4, S6
  - Questa esecuzione e' la baseline Random Forest ECG chest; non include ancora tuning degli iperparametri.


## Passo 12 - ECG chest RF baseline normalization

- Script creato: `src/train_ecg_chest_rf_baseline_norm.py`
- Dataset usato: `data_features/ecg_chest_features_all_60s.csv`
- Assunzione: baseline personale disponibile per ogni soggetto, usando solo finestre con `original_label=1`.
- Modello: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42)`
- Soglia di classificazione: `0.5`.
- Pulizia/scarto finestre: non applicata.
- Feature set testati: `absolute_33`, `baseline_delta_12`, `baseline_zscore_12`, `absolute_plus_delta_45`, `absolute_plus_zscore_45`, `absolute_plus_delta_zscore_57`.
- Migliore feature set: `baseline_zscore_12` con `12` feature.
- Risultati migliori:
  - Accuracy aggregata: 0.8644 (86.44%)
  - Precision aggregata: 0.8308 (83.08%)
  - Recall aggregata: 0.6885 (68.85%)
  - F1 aggregato: 0.7530 (75.30%)
  - Confusion matrix: TN=703, FP=45, FN=100, TP=221
- Delta F1 vs baseline ECG RF precedente: +0.1587 (15.87%)
- File generati:
  - `results/ecg_chest_rf_baseline_norm_results.csv`
  - `results/ecg_chest_rf_baseline_norm_best_per_subject.csv`
  - `results/ecg_chest_rf_baseline_norm_best_feature_importance.csv`
  - `results/ecg_chest_rf_baseline_norm_summary.txt`

## Report HTML ECG chest

- File creato: `results/report_ecg_chest_rf_baseline_norm.html`
- Contenuto: pipeline ECG chest completa, conteggio feature, normalizzazione baseline personale, confronto Random Forest assoluta vs baseline-normalized, confusion matrix, falsi negativi, feature importance e spiegazione pronta per l'orale.

## Passo 13 - Estrazione feature EMG chest

- Script creati:
  - `src/extract_emg_chest_features_all.py`
  - `src/validate_emg_chest_csv.py`
- Segnale usato: `signal["chest"]["EMG"]`, campionato a `700 Hz`.
- Cartella dati usata: `C:\Users\aranh\Downloads\WESAD\WESAD`.
- Segmentazione coerente con BVP/EDA/ECG: finestre di `60 s`, passo di `30 s`, purezza label minima `0.70`.
- Feature estratte: statistiche sul segnale EMG filtrato bandpass `20-250 Hz`, ampiezza rettificata, IEMG/MAV, waveform length, zero crossing, slope sign changes, Willison amplitude, Hjorth e feature spettrali Welch.
- File generati:
  - `data_features/emg_chest_features_all_60s.csv`
- Risultato estrazione:
  - Shape completa: `(1069, 50)`
  - Soggetti: `15`
  - Distribuzione label: `0=748`, `1=321`
  - NaN totali: `0`
- Validazione: `src/validate_emg_chest_csv.py` passata, con finestre e label allineate al CSV BVP di riferimento.


## Passo 14 - EMG chest + Random Forest LOSO

- Script creato: `src/train_emg_chest_rf_loso.py`
- Dataset usato: `data_features/emg_chest_features_all_60s.csv`
- Feature usate: `emg_mean, emg_std, emg_min, emg_max, emg_range, emg_median, emg_iqr, emg_rms, emg_energy, emg_skew, emg_kurtosis, emg_abs_mean, emg_abs_std, emg_abs_median, emg_abs_max, emg_iemg, emg_mav, emg_log_detector, emg_variance, emg_waveform_length, emg_average_amplitude_change, emg_derivative_mean, emg_derivative_std, emg_derivative_max_abs, emg_zero_crossing_count, emg_zero_crossing_rate, emg_slope_sign_change_count, emg_slope_sign_change_rate, emg_willison_amplitude_count, emg_willison_amplitude_rate, emg_hjorth_activity, emg_hjorth_mobility, emg_hjorth_complexity, emg_total_power, emg_mean_frequency, emg_median_frequency, emg_bandpower_20_60, emg_bandpower_60_120, emg_bandpower_120_250, emg_relative_power_20_60, emg_relative_power_60_120, emg_relative_power_120_250`
- Validazione: Leave-One-Subject-Out per soggetto.
- Modello baseline: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42)`
- Scaling: non usato, perche' Random Forest non richiede `StandardScaler`.
- Risultati principali:
  - Accuracy media fold: 0.6741 (67.41%)
  - F1 media fold: 0.3935 (39.35%)
  - Accuracy aggregata globale: 0.6735 (67.35%)
  - F1 aggregata globale: 0.4504 (45.04%)
  - Confusion matrix aggregata: TN=577, FP=171, FN=178, TP=143
- File generati:
  - `results/emg_chest_rf_loso_per_subject.csv`
  - `results/emg_chest_rf_feature_importance.csv`
  - `results/emg_chest_rf_summary.txt`
- Osservazioni:
  - Il modello predice entrambe le classi.
  - Soggetti problematici secondo soglia F1: S10, S2, S5, S6
  - Questa esecuzione e' la baseline Random Forest EMG chest; non include ancora tuning degli iperparametri.

## Passo 15 - EMG chest + Random Forest top15 LOSO

- Script finale: `src/train_emg_chest_rf_top15_loso.py`
- Dataset usato: `data_features/emg_chest_features_all_60s.csv`
- Feature usate: `emg_relative_power_60_120, emg_median, emg_skew, emg_relative_power_20_60, emg_hjorth_complexity, emg_slope_sign_change_count, emg_relative_power_120_250, emg_slope_sign_change_rate, emg_zero_crossing_rate, emg_zero_crossing_count, emg_bandpower_20_60, emg_median_frequency, emg_mean_frequency, emg_hjorth_mobility, emg_bandpower_120_250`
- Validazione: Leave-One-Subject-Out per soggetto.
- Modello: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42)`
- Soglia classificazione: default `0.5`; nessun tuning soglia.
- Iperparametri Random Forest: nessun tuning.
- Risultati:
  - Accuracy aggregata: 0.6978 (69.78%)
  - Precision aggregata: 0.4970 (49.70%)
  - Recall aggregata: 0.5109 (51.09%)
  - F1 aggregato: 0.5038 (50.38%)
  - Confusion matrix: TN=582, FP=166, FN=157, TP=164
- Soggetti problematici secondo soglia F1: S10, S2, S5, S7
- File generati:
  - `results/emg_chest_rf_top15_loso_per_subject.csv`
  - `results/emg_chest_rf_top15_feature_importance.csv`
  - `results/emg_chest_rf_top15_summary.txt`

## Report HTML EMG chest

- File creato: `results/report_emg_chest_rf_top15_loso.html`
- Contenuto: pipeline EMG chest completa, feature estratte, scelta top15, confronto con baseline all-feature, confusion matrix, performance per soggetto, feature importance e conclusione sul ruolo di EMG nel progetto.


## Passo 16 - Estrazione feature RESP chest

- Script creati:
  - `src/extract_resp_chest_features_all.py`
  - `src/validate_resp_chest_csv.py`
- Segnale usato: `signal["chest"]["Resp"]`, campionato a `700 Hz`.
- Cartella dati usata: `C:\Users\aranh\Downloads\WESAD\WESAD`.
- Segmentazione coerente con BVP/EDA/ECG/EMG: finestre di `60 s`, passo di `30 s`, purezza label minima `0.70`.
- Feature estratte: set ridotto non ridondante con statistiche sul segnale RESP filtrato bandpass `0.05-2 Hz`, derivate, rate picchi/trough, breath rate, variabilita' intervalli respiratori, ampiezze ciclo e feature spettrali Welch.
- File generati:
  - `data_features/resp_chest_features_all_60s.csv`
  - `data_features/resp_chest_features_train_s2_s16_60s.csv`
  - `data_features/resp_chest_features_test_s17_60s.csv`
- Risultato estrazione:
  - Shape completa: `(1069, 36)`
  - Soggetti: `15`
  - Distribuzione label: `0=748`, `1=321`
  - NaN totali: `0`
- Validazione: `src/validate_resp_chest_csv.py` passata, con finestre e label allineate al CSV BVP di riferimento.


## Passo 17 - RESP chest + Random Forest LOSO

- Script creato: `src/train_resp_chest_rf_loso.py`
- Dataset usato: `data_features/resp_chest_features_all_60s.csv`
- Feature usate: `resp_mean, resp_std, resp_range, resp_median, resp_iqr, resp_skew, resp_kurtosis, resp_derivative_std, resp_derivative_max_abs, resp_second_derivative_std, resp_peak_rate_per_min, resp_trough_rate_per_min, resp_peak_prominence_std, resp_trough_prominence_std, resp_breath_rate_mean, resp_breath_rate_std, resp_breath_rate_range, resp_interval_rmssd, resp_interval_sdnn, resp_interval_cv, resp_cycle_amplitude_mean, resp_cycle_amplitude_std, resp_dominant_rate_bpm, resp_mean_frequency_hz, resp_bandpower_0_15_0_40, resp_bandpower_0_40_0_75, resp_bandpower_0_75_2_00, resp_relative_power_0_75_2_00`
- Validazione: Leave-One-Subject-Out per soggetto.
- Modello baseline: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42)`
- Scaling: non usato, perche' Random Forest non richiede `StandardScaler`.
- Risultati principali:
  - Accuracy media fold: 0.9019 (90.19%)
  - F1 media fold: 0.8270 (82.70%)
  - Accuracy aggregata globale: 0.9018 (90.18%)
  - F1 aggregata globale: 0.8341 (83.41%)
  - Confusion matrix aggregata: TN=700, FP=48, FN=57, TP=264
- File generati:
  - `results/resp_chest_rf_loso_per_subject.csv`
  - `results/resp_chest_rf_feature_importance.csv`
  - `results/resp_chest_rf_summary.txt`
- Osservazioni:
  - Il modello predice entrambe le classi.
  - Soggetti problematici secondo soglia F1: S4, S6
  - Questa esecuzione e' la baseline Random Forest RESP chest; non include ancora tuning degli iperparametri.

## Report HTML RESP chest

- File creato: `results/report_resp_chest_rf_loso.html`
- Contenuto: pipeline RESP chest completa, scelta del set ridotto non ridondante, validazione CSV, Random Forest LOSO, confronto prima/dopo riduzione feature, confusion matrix, performance per soggetto, feature importance e spiegazione pronta per l'orale.

## Report dettagliato progetto completo

- File creati:
  - `results/report_progetto_wesad_dettagliato.html`
  - `results/report_progetto_wesad_dettagliato.pdf`
- Lunghezza PDF: `14` pagine.
- Contenuto: sintesi del progetto, coerenza con linee guida d'esame, dataset WESAD, mapping label, segmentazione, LOSO, feature extraction per ogni segnale, modelli, normalizzazione, risultati comparativi, focus RESP, analisi critica, confronto con letteratura, traccia per PowerPoint da 15 slide e domande possibili per l'orale.
