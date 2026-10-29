import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# ============================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EXCEL_FILE = BASE_DIR / "data" / "2_Elec" / "elec.xlsx"
MED_FILE = BASE_DIR / "results" / "med_load_actual" / "predictions.csv"

OUTPUT_DIR = BASE_DIR / "results" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("Đang đọc dữ liệu...")

# Đọc dữ liệu phụ tải điện từ sheet x
data = pd.read_excel(
    EXCEL_FILE,
    sheet_name="x",
    index_col=0
)

data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Chỉ lấy biến mục tiêu load_actual
if "load_actual" not in data.columns:
    raise ValueError(
        f"Không tìm thấy cột load_actual. "
        f"Các cột hiện có: {data.columns.tolist()}"
    )

series = pd.to_numeric(
    data["load_actual"],
    errors="coerce"
).dropna()

# Đọc kết quả dự báo MED hiện có
med = pd.read_csv(MED_FILE)
med["date"] = pd.to_datetime(med["date"])

med = med.sort_values("date").reset_index(drop=True)

# Kiểm tra các cột cần thiết
required_columns = {"date", "actual", "prediction"}

if not required_columns.issubset(med.columns):
    raise ValueError(
        f"File MED cần có các cột: {required_columns}"
    )

if med["date"].duplicated().any():
    raise ValueError("File MED có timestamp bị trùng.")

if med[["actual", "prediction"]].isna().any().any():
    raise ValueError("File MED có giá trị actual/prediction bị thiếu.")


# ============================================================
# 3. XÁC ĐỊNH TRAIN VÀ TEST
# ============================================================

# Lấy các timestamp trong tập test MED
test_dates = pd.DatetimeIndex(med["date"])

# Kiểm tra các timestamp có trong Excel không
missing_dates = test_dates.difference(series.index)

if len(missing_dates) > 0:
    raise ValueError(
        f"Có {len(missing_dates)} timestamp trong MED "
        f"không tìm thấy trong Excel. Ví dụ: "
        f"{missing_dates[:5].tolist()}"
    )

# Lấy giá trị thực tế theo đúng timestamp của MED
actual = series.reindex(test_dates)

# Kiểm tra giá trị thực tế có khớp với MED không
med_actual = med["actual"].to_numpy()
excel_actual = actual.to_numpy()

if not np.allclose(
    med_actual,
    excel_actual,
    rtol=1e-5,
    atol=1e-3
):
    raise ValueError(
        "Giá trị actual trong MED không khớp với Excel. "
        "Không nên so sánh trước khi xác định nguyên nhân."
    )

# Tập train gồm các mốc trước thời điểm test đầu tiên
train = series.loc[series.index < test_dates[0]]

if len(train) == 0:
    raise ValueError("Tập train rỗng.")

if not train.index.is_monotonic_increasing:
    raise ValueError("Dữ liệu train chưa được sắp xếp theo thời gian.")

print(f"Số mẫu train: {len(train)}")
print(f"Số mẫu test: {len(test_dates)}")
print(f"Test bắt đầu: {test_dates[0]}")
print(f"Test kết thúc: {test_dates[-1]}")


# ============================================================
# 4. MÔ HÌNH NAIVE
# ============================================================

print("\nĐang chạy Naive...")

# Dự báo một bước:
# Giá trị dự báo tại thời điểm t bằng giá trị thực tế tại t-1.
# Với chu kỳ 6 giờ, mỗi bước tương ứng 6 giờ.

full_series = series.reindex(
    series.index.union(test_dates)
).sort_index()

naive_predictions = []

for date in test_dates:
    previous_dates = full_series.index[
        full_series.index < date
    ]

    if len(previous_dates) == 0:
        raise ValueError(
            f"Không có dữ liệu trước thời điểm {date}"
        )

    previous_date = previous_dates[-1]
    naive_predictions.append(full_series.loc[previous_date])

naive_predictions = np.asarray(naive_predictions)


# ============================================================
# 5. MÔ HÌNH ARIMA
# ============================================================

print("\nĐang huấn luyện ARIMA(5,1,0)...")

# ARIMA được huấn luyện ban đầu trên tập train.
# Sau mỗi bước dự báo, mô hình được cập nhật bằng giá trị
# thực tế vừa quan sát, không huấn luyện lại toàn bộ từ đầu.

arima_model = ARIMA(
    train.astype(float),
    order=(5, 1, 0)
)

arima_result = arima_model.fit()

arima_predictions = []

history_model = arima_result

for i, date in enumerate(test_dates):
    # Dự báo một bước tiếp theo
    forecast = history_model.forecast(steps=1)
    prediction = float(forecast.iloc[0])

    arima_predictions.append(prediction)

    # Cập nhật bằng giá trị thực tế tại timestamp hiện tại
    # để dự báo timestamp tiếp theo.
    if i < len(test_dates) - 1:
        observed_value = float(actual.iloc[i])

        history_model = history_model.append(
            [observed_value],
            refit=False
        )

    if (i + 1) % 100 == 0:
        print(f"ARIMA đã dự báo {i + 1}/{len(test_dates)} mẫu.")

arima_predictions = np.asarray(arima_predictions)


# ============================================================
# 6. TỔNG HỢP DỰ BÁO
# ============================================================

predictions = pd.DataFrame({
    "date": test_dates,
    "actual": actual.to_numpy(),
    "Naive": naive_predictions,
    "ARIMA": arima_predictions,
    "MED": med["prediction"].to_numpy()
})

predictions.to_csv(
    OUTPUT_DIR / "predictions_comparison.csv",
    index=False
)


# ============================================================
# 7. TÍNH CÁC CHỈ SỐ ĐÁNH GIÁ
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    # Bỏ qua các điểm actual bằng 0 khi tính MAPE
    nonzero = y_true != 0

    if np.any(nonzero):
        mape = np.mean(
            np.abs(error[nonzero] / y_true[nonzero])
        ) * 100
    else:
        mape = np.nan

    denominator = np.abs(y_true) + np.abs(y_pred)

    valid = denominator != 0

    if np.any(valid):
        smape = np.mean(
            2 * np.abs(error[valid]) / denominator[valid]
        ) * 100
    else:
        smape = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "MAPE (%)": mape,
        "sMAPE (%)": smape
    }


# Tất cả mô hình được đánh giá trên cùng các mẫu
results = []

for model_name in ["Naive", "ARIMA", "MED"]:
    metrics = calculate_metrics(
        predictions["actual"],
        predictions[model_name]
    )

    metrics["Model"] = model_name
    results.append(metrics)

metrics_df = pd.DataFrame(results)

metrics_df = metrics_df[
    ["Model", "MAE", "RMSE", "MSE", "MAPE (%)", "sMAPE (%)"]
]

metrics_df.to_csv(
    OUTPUT_DIR / "metrics_comparison.csv",
    index=False
)


# ============================================================
# 8. VẼ BIỂU ĐỒ
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    predictions["date"],
    predictions["actual"],
    label="Actual",
    linewidth=2
)

plt.plot(
    predictions["date"],
    predictions["Naive"],
    label="Naive",
    alpha=0.8
)

plt.plot(
    predictions["date"],
    predictions["ARIMA"],
    label="ARIMA",
    alpha=0.8
)

plt.plot(
    predictions["date"],
    predictions["MED"],
    label="MED",
    alpha=0.8
)

plt.title("Actual vs Naive vs ARIMA vs MED")
plt.xlabel("Time")
plt.ylabel("Electricity load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "prediction_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. IN KẾT QUẢ
# ============================================================

print("\n" + "=" * 75)
print("KẾT QUẢ SO SÁNH")
print("=" * 75)

print(metrics_df.to_string(index=False))

print("\nĐã lưu kết quả tại:")
print(OUTPUT_DIR / "metrics_comparison.csv")
print(OUTPUT_DIR / "predictions_comparison.csv")
print(OUTPUT_DIR / "prediction_comparison.png")