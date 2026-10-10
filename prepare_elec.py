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
# 3. XỬ LÝ THỜI GIAN
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - PROCESS DATETIME")
print("=" * 70)

# utc=True xử lý được các timezone khác nhau trong dữ liệu
energy["time"] = pd.to_datetime(
    energy["time"],
    utc=True
)

weather["dt_iso"] = pd.to_datetime(
    weather["dt_iso"],
    utc=True
)

# Chuyển về timezone của Tây Ban Nha.
# Spain sử dụng CET/CEST và pandas sẽ tự xử lý DST.
energy["time"] = (
    energy["time"]
    .dt.tz_convert("Europe/Madrid")
    .dt.tz_localize(None)
)

weather["dt_iso"] = (
    weather["dt_iso"]
    .dt.tz_convert("Europe/Madrid")
    .dt.tz_localize(None)
)

print("Energy time:")
print(energy["time"].min(), "->", energy["time"].max())

print("Weather time:")
print(weather["dt_iso"].min(), "->", weather["dt_iso"].max())


# ============================================================
# 4. XỬ LÝ TÊN THÀNH PHỐ
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
# 5. CHỌN 10 BIẾN THỜI TIẾT
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
# 6. TẠO 10 CỘT WEATHER
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - CREATE 10 WEATHER FEATURES")
print("=" * 70)

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
# 7. XỬ LÝ MISSING WEATHER
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
weather_wide = weather_wide.set_index("dt_iso")

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
# 8. XỬ LÝ ELECTRICITY TARGET
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
# 9. HOURLY → 6-HOURLY
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - CONVERT HOURLY TARGET TO 6-HOURLY")
print("=" * 70)

# Trung bình mỗi 6 giờ
target_6h = (
    target_hourly[target_col]
    .resample("6h")
    .mean()
    .to_frame()
)

# Đổi tên theo config của repo
target_6h = target_6h.rename(
    columns={
        target_col: "load_actual"
    }
)

print("6-hourly target shape:", target_6h.shape)

print("\nFirst 5 target values:")
print(target_6h.head())


# ============================================================
# 10. CHUẨN BỊ SHEET x
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
# 11. CHUẨN BỊ SHEET y
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
# 12. TẠO THƯ MỤC
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 13. GHI EXCEL
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
# 14. KIỂM TRA CUỐI
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