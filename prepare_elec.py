import pandas as pd
from pathlib import Path


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENERGY_FILE = BASE_DIR / "energy_dataset.csv"
WEATHER_FILE = BASE_DIR / "weather_features.csv"

OUTPUT_DIR = BASE_DIR / "data" / "2_Elec"
OUTPUT_FILE = OUTPUT_DIR / "elec.xlsx"


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("STEP 1 - READ DATA")
print("=" * 70)

print("Reading energy dataset...")
energy = pd.read_csv(ENERGY_FILE)

print("Reading weather dataset...")
weather = pd.read_csv(WEATHER_FILE)

print("Energy shape :", energy.shape)
print("Weather shape:", weather.shape)


# ============================================================
# 3. LÀM SẠCH DỮ LIỆU NĂNG LƯỢNG (TỪ KAGGLE)
# ============================================================

print("\n" + "=" * 70)
print("STEP 1.1 - CLEAN ENERGY DATASET (KAGGLE METHOD)")
print("=" * 70)

# 1. Loại bỏ các cột chỉ chứa toàn giá trị 0 hoặc NaN
cols_to_drop = [col for col in energy.columns if energy[col].nunique() <= 1 or energy[col].isna().all()]
if cols_to_drop:
    print(f"Dropping constant/all-NaN columns: {cols_to_drop}")
    energy = energy.drop(columns=cols_to_drop)

# 2. Loại bỏ các cột dự báo day-ahead forecasts (nếu có trong bảng)
day_ahead_cols = [col for col in energy.columns if "day-ahead" in col.lower()]
if day_ahead_cols:
    print(f"Dropping day-ahead forecast columns: {day_ahead_cols}")
    energy = energy.drop(columns=day_ahead_cols)

print("Energy shape after cleaning:", energy.shape)


# ============================================================
# 4. XỬ LÝ THỜI GIAN
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - PROCESS DATETIME")
print("=" * 70)

# utc=True xử lý được các timezone khác nhau trong dữ liệu
# Chuẩn hóa cả hai chuỗi về UTC.
# Loại bỏ thông tin timezone sau khi đã chuẩn hóa,
# nhưng vẫn giữ nguyên thời gian UTC.

energy["time"] = pd.to_datetime(
    energy["time"],
    utc=True
).dt.tz_localize(None)

weather["dt_iso"] = pd.to_datetime(
    weather["dt_iso"],
    utc=True
).dt.tz_localize(None)

# Chuyển về timezone của Tây Ban Nha.
# Spain sử dụng CET/CEST và pandas sẽ tự xử lý DST.


print("Energy time:")
print(energy["time"].min(), "->", energy["time"].max())

print("Weather time:")
print(weather["dt_iso"].min(), "->", weather["dt_iso"].max())


# ============================================================
# 5. XỬ LÝ TÊN THÀNH PHỐ
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CLEAN CITY NAMES")
print("=" * 70)

weather["city_name"] = weather["city_name"].str.strip()

cities = [
    "Madrid",
    "Barcelona",
    "Valencia",
    "Seville",
    "Bilbao"
]

print("Cities used:")
for city in cities:
    print("-", city)


# ============================================================
# 6. CHỌN 10 BIẾN THỜI TIẾT
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - SELECT WEATHER FEATURES")
print("=" * 70)

weather_selected = weather[
    weather["city_name"].isin(cities)
][
    ["dt_iso", "city_name", "temp", "humidity"]
].copy()

print("Selected weather shape:", weather_selected.shape)


# ============================================================
# 7. TẠO 10 CỘT WEATHER
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - CREATE 10 WEATHER FEATURES")
print("=" * 70)

# Kiểm tra bản ghi trùng theo thời gian và thành phố
duplicate_weather = weather_selected.duplicated(
    subset=["dt_iso", "city_name"]
).sum()

print("Weather duplicate records:", duplicate_weather)

weather_wide = weather_selected.pivot_table(
    index="dt_iso",
    columns="city_name",
    values=["temp", "humidity"],
    aggfunc="mean"
)

weather_wide.columns = [
    f"{feature}_{city}"
    for feature, city in weather_wide.columns
]

weather_wide = weather_wide.reset_index()

weather_columns = []

for city in cities:
    weather_columns.append(f"temp_{city}")
    weather_columns.append(f"humidity_{city}")

weather_wide = weather_wide[
    ["dt_iso"] + weather_columns
]

print("Weather columns:")
for col in weather_columns:
    print("-", col)

print("Weather wide shape:", weather_wide.shape)


# ============================================================
# 8. XỬ LÝ MISSING WEATHER
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - CHECK WEATHER MISSING VALUES")
print("=" * 70)

weather_missing = weather_wide[weather_columns].isna().sum()

print(weather_missing)

print(
    "\nTotal weather missing values:",
    weather_missing.sum()
)

# Nội suy theo thời gian
weather_wide = weather_wide.set_index("dt_iso").sort_index()

# Tạo lưới thời gian UTC liên tục theo giờ
full_weather_index = pd.date_range(
    start=weather_wide.index.min(),
    end=weather_wide.index.max(),
    freq="1h"
)

# Đưa dữ liệu về lưới thời gian đầy đủ
weather_wide = weather_wide.reindex(full_weather_index)
weather_wide.index.name = "dt_iso"

print(
    "Weather missing before interpolation:",
    weather_wide[weather_columns].isna().sum().sum()
)

# Nội suy các giá trị khí tượng bị thiếu
weather_wide[weather_columns] = (
    weather_wide[weather_columns]
    .interpolate(method="time")
    .ffill()
    .bfill()
)

weather_wide = weather_wide.reset_index()

print(
    "Missing weather after interpolation:",
    weather_wide[weather_columns].isna().sum().sum()
)


# ============================================================
# 9. XỬ LÝ ELECTRICITY TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - PROCESS ELECTRICITY TARGET")
print("=" * 70)

target_col = "total load actual"

print("Original missing values in target:")
print(energy[target_col].isna().sum())

target_hourly = energy[
    ["time", target_col]
].copy()

target_hourly = target_hourly.sort_values("time")

target_hourly = target_hourly.set_index("time")

# Nội suy missing values
target_hourly[target_col] = (
    target_hourly[target_col]
    .interpolate(method="time")
    .ffill()
    .bfill()
)

print(
    "Missing target after interpolation:",
    target_hourly[target_col].isna().sum()
)



# ============================================================
# 10. HOURLY → 6-HOURLY
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - CONVERT HOURLY TARGET TO 6-HOURLY")
print("=" * 70)

# Sắp xếp dữ liệu theo thời gian
target_hourly = target_hourly.sort_index()

# Kiểm tra timestamp trùng
if target_hourly.index.has_duplicates:
    raise ValueError("Target hourly có timestamp bị trùng.")

# Kiểm tra số quan sát trong từng khoảng 6 giờ.
# origin="start" căn khoảng tổng hợp từ timestamp đầu tiên.
target_6h_stats = (
    target_hourly[target_col]
    .resample("6h", origin="start")
    .agg(["mean", "count"])
)

print("Số lượng quan sát trong mỗi khoảng 6 giờ:")
print(target_6h_stats["count"].value_counts().sort_index())

# Không chấp nhận các khoảng có ít hoặc nhiều hơn 6 quan sát
invalid_bins = target_6h_stats[
    target_6h_stats["count"] != 6
]

if not invalid_bins.empty:
    print("\nCác khoảng 6 giờ không đủ 6 quan sát:")
    print(invalid_bins.head(10))

    raise ValueError(
        "Có khoảng tổng hợp không chứa đúng 6 quan sát. "
        "Cần kiểm tra dữ liệu hourly trước khi tiếp tục."
    )

# Lấy giá trị trung bình của 6 quan sát trong mỗi khoảng
target_6h = (
    target_6h_stats[["mean"]]
    .rename(columns={"mean": "load_actual"})
)

print("\n6-hourly target shape:", target_6h.shape)

print("\nFirst 5 target values:")
print(target_6h.head())

print("\nLast 5 target values:")
print(target_6h.tail())



# ============================================================
# 11. CHUẨN BỊ SHEET x
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - PREPARE SHEET X")
print("=" * 70)

target_6h = target_6h.reset_index()

target_6h = target_6h.sort_values("time")

x = target_6h[
    ["time", "load_actual"]
].copy()

x["time"] = x["time"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

x = x.set_index("time")

print("X shape:", x.shape)
print("X columns:", list(x.columns))


# ============================================================
# 12. CHUẨN BỊ SHEET y
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - PREPARE SHEET Y")
print("=" * 70)

weather_wide = weather_wide.sort_values("dt_iso")

y = weather_wide[
    ["dt_iso"] + weather_columns
].copy()

y["dt_iso"] = y["dt_iso"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

y = y.set_index("dt_iso")

print("Y shape:", y.shape)
print("Y columns:", list(y.columns))


# ============================================================
# 13. TẠO THƯ MỤC
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 14. GHI EXCEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 - SAVE EXCEL")
print("=" * 70)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    x.to_excel(
        writer,
        sheet_name="x"
    )

    y.to_excel(
        writer,
        sheet_name="y"
    )


# ============================================================
# 15. KIỂM TRA CUỐI
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 - FINAL CHECK")
print("=" * 70)

print("Output file:")
print(OUTPUT_FILE)

print("\nSheet x:")
print("Shape:", x.shape)
print(x.head())

print("\nSheet y:")
print("Shape:", y.shape)
print(y.head())

print("\nMissing values in x:")
print(x.isna().sum())

print("\nMissing values in y:")
print(y.isna().sum())

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

print("File created successfully:")
print(OUTPUT_FILE)