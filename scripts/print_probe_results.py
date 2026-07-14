import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_link_prediction import prepare_data, timestamp_to_float
import logging
from dyfo.config import DyFOConfig, DataConfig

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    spearman = spearmanr(y_true_flat, y_pred_flat).statistic
    return {"R2": r2, "MAE": mae, "Spearman": spearman}

def main():
    UNIVERSE = [
        'AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM', 
        'PG', 'GOOGL', 'META', 'TSLA', 'PFE', 
        'V', 'HD', 'CVX', 'ABBV', 'KO'
    ]

    start_date = "2016-01-01"
    end_date = "2023-06-01"
    
    logger = logging.getLogger("dyfo_probe2")
    logger.setLevel(logging.CRITICAL)
    
    config = DyFOConfig()
    data_config = DataConfig()
    
    # This might have to redownload data or use cache if available.
    data = prepare_data(
        tickers=UNIVERSE,
        start=start_date,
        end=end_date,
        benchmark="SPY",
        config=config,
        data_config=data_config,
        logger=logger
    )
    
    dates = sorted(list(data["corr_labels_by_date"].keys()))
    
    calm_date_target = timestamp_to_float(pd.Timestamp("2019-07-01"))
    calm_test_start = next(i for i, d in enumerate(dates) if d >= calm_date_target)
    calm_test_dates = dates[calm_test_start : calm_test_start + 125]
    calm_val_dates = dates[calm_test_start - 125 : calm_test_start]
    calm_train_dates = dates[calm_test_start - 125 - 500 : calm_test_start - 125]
    
    break_date_target = timestamp_to_float(pd.Timestamp("2022-01-01"))
    break_test_start = next(i for i, d in enumerate(dates) if d >= break_date_target)
    break_test_dates = dates[break_test_start : break_test_start + 125]
    break_val_dates = dates[break_test_start - 125 : break_test_start]
    break_train_dates = dates[break_test_start - 125 - 500 : break_test_start - 125]
    
    windows = {
        "CALM": {
            "train": calm_train_dates,
            "val": calm_val_dates,
            "test": calm_test_dates
        },
        "BREAK": {
            "train": break_train_dates,
            "val": break_val_dates,
            "test": break_test_dates
        }
    }
    
    results = {}
    num_pairs = len(UNIVERSE) * (len(UNIVERSE) - 1) // 2
    pairs = []
    for i in range(len(UNIVERSE)):
        for j in range(i+1, len(UNIVERSE)):
            pairs.append((i, j))
            
    for w_name, w_dates in windows.items():
        all_window_dates = w_dates['train'] + w_dates['val'] + w_dates['test']
        y_all = np.zeros((len(all_window_dates), num_pairs))
        for t_idx, d in enumerate(all_window_dates):
            corr_matrix = data["corr_labels_by_date"].get(d, {})
            for p_idx, (s, dst) in enumerate(pairs):
                y_all[t_idx, p_idx] = corr_matrix.get((s, dst), corr_matrix.get((dst, s), 0.0))
        
        test_start_idx = len(w_dates['train']) + len(w_dates['val'])
        test_indices = list(range(test_start_idx, len(all_window_dates)))
        
        y_test = y_all[test_indices]
        
        # Persistence
        y_pred_pers = y_all[[i-1 for i in test_indices]]
        metrics_pers = compute_metrics(y_test, y_pred_pers)
        
        # EWMA
        best_ewma_metrics = None
        best_lam = None
        for lam in [0.80, 0.90, 0.94, 0.97]:
            s = np.zeros_like(y_all)
            s[0] = y_all[0]
            for t in range(1, len(all_window_dates)):
                s[t] = lam * y_all[t] + (1 - lam) * s[t-1]
            
            y_pred_ewma = s[[i-1 for i in test_indices]]
            m = compute_metrics(y_test, y_pred_ewma)
            if best_ewma_metrics is None or m["R2"] > best_ewma_metrics["R2"]:
                best_ewma_metrics = m
                best_lam = lam
                
        # DyFO (Read from CSV)
        preds_path = f"D:/projetos/DyFO/results/probe_{w_name}_dyfo_preds.csv"
        df_dyfo = pd.read_csv(preds_path)
        # Sort values properly or map them
        y_pred_dyfo = np.zeros_like(y_test)
        
        for t_idx, d in enumerate(w_dates['test']):
            df_day = df_dyfo[df_dyfo['date'] == d]
            # Since node ids are strings but in output they are 0-14, let's just create a matrix
            for _, row in df_day.iterrows():
                src, dst = int(row['src']), int(row['dst'])
                # Find pair index
                if src > dst: src, dst = dst, src
                try:
                    p_idx = pairs.index((src, dst))
                    y_pred_dyfo[t_idx, p_idx] = row['pred']
                except ValueError:
                    pass
                    
        metrics_dyfo = compute_metrics(y_test, y_pred_dyfo)
                
        results[w_name] = {
            "Persistence": metrics_pers,
            f"EWMA(lam={best_lam})": best_ewma_metrics,
            "DyFO": metrics_dyfo
        }
        
    # Write report
    report = "Diagnostic Probe Results\n========================\n\n"
    for w_name in ["CALM", "BREAK"]:
        report += f"Window: {w_name}\n"
        report += f"{'Model':<15} {'R2':<8} {'MAE':<8} {'Spearman':<8}\n"
        res = results[w_name]
        for model_name, metrics in res.items():
            report += f"{model_name:<15} {metrics['R2']:>8.4f} {metrics['MAE']:>8.4f} {metrics['Spearman']:>8.4f}\n"
        report += "\n"
        
    report += "DECISION SIGNAL:\n"
    report += "----------------\n"
    
    def check_beat(dyfo_r2, baseline_r2):
        if dyfo_r2 > baseline_r2 + 0.02: return "beats"
        if dyfo_r2 < baseline_r2 - 0.02: return "loses to"
        return "matches"

    for w_name in ["CALM", "BREAK"]:
        res = results[w_name]
        dyfo_r2 = res["DyFO"]["R2"]
        pers_r2 = res["Persistence"]["R2"]
        best_ewma_key = next(k for k in res.keys() if "EWMA" in k)
        ewma_r2 = res[best_ewma_key]["R2"]
        best_baseline = max(pers_r2, ewma_r2)
        headroom = dyfo_r2 - best_baseline
        
        beat_str = check_beat(dyfo_r2, best_baseline)
        report += f"- In the {w_name} window, DyFO {beat_str} persistence & EWMA.\n"
        report += f"  Headroom (DyFO R2 - best naive R2): {headroom:+.4f}\n"
        
    report += "\nInterpretation note:\n"
    report += "Persistence/EWMA use the lagged target rho^t, which DyFO deliberately excludes.\n"
    report += "These are diagnostic references, not competing baselines. The relevant question\n"
    report += "is whether DyFO's causal forecast matches naive autocorrelation in calm regimes\n"
    report += "while degrading less in the break regime.\n"
    
    with open("D:/projetos/DyFO/results/probe_results.txt", "w", encoding="utf-8") as f:
        f.write(report)
        
    print(report)

if __name__ == "__main__":
    main()
