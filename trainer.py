#!/usr/bin/env python
"""model_trainer.py

Top-level orchestration script that wires data, the MED model wrapper and
training / evaluation loops together.

It supports
* K-fold training (delegated to :class:`med_model.MEDModel`)
* test-time evaluation and plotting
* metric logging + Excel dump via :pydata:`pandas`

Author : Jiaxi Liu
License: MIT
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from models.med_model import MEDModel
from utils import (
    fix_random_seed,
    plot_predictions_and_real_values,
    symmetric_mean_absolute_percentage_error,
)

# -----------------------------------------------------------------------------

MetricDict = Dict[str, float]


class ModelTrainer:
    """End-to-end trainer for MED models (dual-frequency version)."""

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        train_data: Tuple[pd.DataFrame, pd.DataFrame],
        test_data: Tuple[pd.DataFrame, pd.DataFrame],
        args,
    ) -> None:
        fix_random_seed(args.seed)
        self.args = args
        self.train_data = train_data
        self.test_data = test_data

        # -------------------------------------------------------------- #
        self.model = MEDModel(args)
        n_param = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"Model params: {n_param / 1e6:.2f} M")

        self.key_var = args.key_var
        self.metrics: Dict[str, MetricDict] = {}
        self.real_series: Dict[str, str] = {}
        self.pred_series: Dict[str, str] = {}

    # ------------------------------------------------------------------
    #  TRAIN + TEST
    # ------------------------------------------------------------------
    def train(self) -> None:
        """K-fold training via :class:`MEDModel`."""
        self.model.train(self.train_data)

    def test(self) -> None:
        """Generate predictions, evaluate metrics, and save results."""

        # 1. Du doan tren tap test
        preds = np.asarray(
            self.model.test(self.test_data, self.train_data)
        ).ravel()

        # 2. Lay gia tri thuc te
        target_df = self.test_data[1]
        reals = target_df[self.key_var].to_numpy().ravel()

        # 3. Kiem tra so luong du doan va gia tri thuc te
        if len(preds) != len(reals):
            raise ValueError(
                f"Prediction length ({len(preds)}) != "
                f"actual length ({len(reals)})"
            )

        # 4. Tinh cac chi so danh gia
        mae = mean_absolute_error(reals, preds)
        mape = mean_absolute_percentage_error(reals, preds)
        mse = mean_squared_error(reals, preds)
        smape = symmetric_mean_absolute_percentage_error(
            reals, preds
        )

        # 5. Luu cac chi so vao bo nho
        self.metrics[self.key_var] = {
            "mae": float(mae),
            "mape": float(mape),
            "mse": float(mse),
            "smape": float(smape),
        }

        self.real_series[self.key_var] = ",".join(map(str, reals))
        self.pred_series[self.key_var] = ",".join(map(str, preds))

        # 6. Tao thu muc luu ket qua
        result_dir = (
            Path("results") /
            f"{self.args.model_type}_{self.key_var}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        # 7. Tao bang du doan
        result_df = pd.DataFrame({
            "actual": reals,
            "prediction": preds,
            "error": preds - reals,
            "absolute_error": np.abs(preds - reals),
        })

        # 8. Them cot thoi gian neu co
        date_columns = [
            "date", "Date", "datetime", "Datetime",
            "timestamp", "Timestamp", "time", "Time",
        ]

        date_col = next(
            (col for col in date_columns if col in target_df.columns),
            None,
        )

        if date_col is not None:
            result_df.insert(
                0, "date", target_df[date_col].to_numpy()
            )
        elif isinstance(target_df.index, pd.DatetimeIndex):
            result_df.insert(
                0, "date", target_df.index.to_numpy()
            )

        # 9. Luu du doan
        prediction_file = result_dir / "predictions.csv"
        result_df.to_csv(
            prediction_file,
            index=False,
            encoding="utf-8-sig",
        )

        # 10. Luu cac chi so danh gia
        metrics_file = result_dir / "metrics.csv"
        pd.DataFrame(
            [self.metrics[self.key_var]]
        ).to_csv(
            metrics_file,
            index=False,
            encoding="utf-8-sig",
        )

        # 11. Ve bieu do neu duoc bat trong config
        if self.args.plot:
            plot_predictions_and_real_values(
                preds,
                reals,
                title=self.args.model_type,
                save_dir=str(result_dir),
            )

        # 12. In ket qua
        print(
            f"MAE={mae:.4f} | "
            f"MAPE={mape:.4f} | "
            f"MSE={mse:.4f} | "
            f"sMAPE={smape:.4f}"
        )

        print(f"Predictions saved to: {prediction_file}")
        print(f"Metrics saved to: {metrics_file}")