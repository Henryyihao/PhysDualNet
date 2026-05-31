import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run_all_predict_plots(nino_pred, future_dates, save_dir, title_suffix="", obs_nino34=None, obs_dates=None):
    os.makedirs(save_dir, exist_ok=True)
    pred = np.asarray(nino_pred, dtype=float)
    dates = pd.to_datetime(future_dates)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "nino34_pred": pred}).to_csv(
        os.path.join(save_dir, "future_nino34_prediction.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(dates, pred, marker="o", label="Prediction")
    if obs_nino34 is not None and obs_dates is not None and len(obs_nino34) > 0:
        obs_dates = pd.to_datetime(obs_dates)
        obs = np.asarray(obs_nino34, dtype=float)
        pd.DataFrame({"date": obs_dates.strftime("%Y-%m-%d"), "nino34_obs": obs}).to_csv(
            os.path.join(save_dir, "future_nino34_observed_overlap.csv"), index=False
        )
        ax.plot(obs_dates, obs, marker="s", label="Observed")
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Nino3.4 Forecast {title_suffix}".strip())
    ax.set_xlabel("Date")
    ax.set_ylabel("Nino3.4 anomaly")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "future_nino34_prediction.png"))
    plt.close(fig)
