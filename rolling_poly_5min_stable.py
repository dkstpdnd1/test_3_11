import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 0. 설정
# =========================================
INPUT_CSV = "area_count_time_full_3.csv"

# 예측 설정
WINDOW_MINUTES = 5
FORECAST_MINUTES = 5
CANDIDATE_DEGREES = [1, 2, 3]
VALIDATION_MINUTES = 2

# smoothing
USE_SMOOTHING = True
SMOOTHING_WINDOW = 3

# 혼잡도 변환
W_SCALE = 37.3

# 안정화 제한
USE_LOCAL_CAP = True          # 직전 window 기준 상한/하한 제한
LOCAL_CAP_UP = 1.30           # 직전 5분 최대값의 130% 초과 금지
LOCAL_CAP_DOWN = 0.70         # 직전 5분 최소값의 70% 미만 방지(최소값이 0이면 별도 처리)

USE_DELTA_LIMIT = True        # 마지막 값 기준 급격한 변화량 제한
DELTA_LIMIT_ABS = 5.0         # 분당 사람 수 기준 절대 변화량 제한

USE_NONNEGATIVE = True        # 음수 제거

# 출력 파일
OUTPUT_PRED_CSV = "rolling_poly_5min_stable_prediction.csv"
OUTPUT_METRICS_CSV = "rolling_poly_5min_stable_metrics.csv"
OUTPUT_DEGREE_CSV = "rolling_poly_5min_stable_selected_degree_stats.csv"
OUTPUT_DEGREE_LOG_CSV = "rolling_poly_5min_stable_degree_log.csv"
GRAPH_DIR = "rolling_poly_5min_stable_area_graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)

# =========================================
# 1. 데이터 로드
# =========================================
df = pd.read_csv(INPUT_CSV)
df["area"] = df["area"].astype(str).str.strip()

# 10초 단위 -> 1분 단위 평균
df["minute_index"] = ((df["time_index"] - 1) // 6).astype(int)

area_counts = (
    df.groupby(["minute_index", "area"])["num_people"]
    .mean()
    .reset_index(name="num_people")
)

pivot_df = area_counts.pivot(
    index="minute_index",
    columns="area",
    values="num_people"
).fillna(0)

pivot_df = pivot_df.sort_index()

if USE_SMOOTHING:
    pivot_df = pivot_df.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

# =========================================
# 2. 구역별 처리 파라미터
# =========================================
def get_service_params(area: str):
    area = str(area).strip()

    if area in ['A', 'C', 'D', 'E', 'H', 'J', 'K', 'M', 'N']:
        servers = 10
        Ts = 4
    elif area in ['B', 'F', 'G', 'L']:
        servers = 10
        Ts = 5
    elif area in ['IM1', 'IM2']:
        servers = 10
        Ts = 4
    else:
        # Great Hall, Outside 등
        servers = 1000
        Ts = 0.1

    return servers, Ts

def people_to_waiting_time(num_people, area):
    servers, Ts = get_service_params(area)
    return np.maximum(0, (num_people - servers) * Ts / servers)

def waiting_to_congestion(waiting_time, w_scale=W_SCALE):
    return 1 - np.exp(-waiting_time / w_scale)

# =========================================
# 3. 평가 함수
# =========================================
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# =========================================
# 4. 다항식 적합 / 예측
# =========================================
def fit_poly_predict(x_train, y_train, x_target, degree):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_target = np.asarray(x_target, dtype=float)

    x0 = x_train[0]
    x_train_norm = x_train - x0
    x_target_norm = x_target - x0

    effective_degree = min(degree, len(x_train_norm) - 1)

    coeffs = np.polyfit(x_train_norm, y_train, effective_degree)
    poly = np.poly1d(coeffs)

    y_pred = poly(x_target_norm)
    return y_pred, effective_degree, coeffs

def choose_best_degree_short_window(t_window, y_window, candidate_degrees, validation_minutes):
    """
    짧은 window(5분)에서 차수 자동 선택
    마지막 validation_minutes를 검증으로 사용
    """
    n = len(t_window)

    if n <= validation_minutes + 1:
        best_degree = min(candidate_degrees)
        return best_degree, {
            "fallback": True,
            "scores": []
        }

    split_idx = n - validation_minutes
    x_train = t_window[:split_idx]
    y_train = y_window[:split_idx]
    x_val = t_window[split_idx:]
    y_val = y_window[split_idx:]

    scores = []

    for deg in candidate_degrees:
        pred_val, eff_deg, _ = fit_poly_predict(x_train, y_train, x_val, deg)
        pred_val = np.maximum(0, pred_val)

        raw_mae = mae(y_val, pred_val)

        # 짧은 구간에서는 고차수에 약한 패널티 부여
        score_with_penalty = raw_mae + 0.01 * deg

        scores.append({
            "degree": deg,
            "effective_degree": eff_deg,
            "val_mae": raw_mae,
            "score_with_penalty": score_with_penalty
        })

    scores = sorted(scores, key=lambda x: (x["score_with_penalty"], x["degree"]))
    best = scores[0]

    return best["degree"], {
        "fallback": False,
        "scores": scores
    }

# =========================================
# 5. 안정화 함수
# =========================================
def stabilize_prediction(raw_pred, y_window):
    """
    급격한 튐 방지 제한
    순서:
    1) 음수 제거
    2) local cap
    3) last value 기준 delta limit
    """
    pred = float(raw_pred)
    y_window = np.asarray(y_window, dtype=float)

    if USE_NONNEGATIVE:
        pred = max(0.0, pred)

    local_min = float(np.min(y_window))
    local_max = float(np.max(y_window))
    last_value = float(y_window[-1])

    # 1) local cap
    if USE_LOCAL_CAP:
        upper_bound = local_max * LOCAL_CAP_UP

        if local_min > 0:
            lower_bound = local_min * LOCAL_CAP_DOWN
        else:
            lower_bound = 0.0

        pred = np.clip(pred, lower_bound, upper_bound)

    # 2) last value 기준 급변 제한
    if USE_DELTA_LIMIT:
        pred = np.clip(
            pred,
            last_value - DELTA_LIMIT_ABS,
            last_value + DELTA_LIMIT_ABS
        )

    if USE_NONNEGATIVE:
        pred = max(0.0, pred)

    return pred

# =========================================
# 6. Rolling Polynomial Extrapolation
# =========================================
results = []
degree_logs = []

minute_indices = pivot_df.index.to_numpy(dtype=float)
areas = pivot_df.columns.tolist()

for area in areas:
    y_all = pivot_df[area].to_numpy(dtype=float)
    t_all = minute_indices

    for current_pos in range(WINDOW_MINUTES - 1, len(t_all) - FORECAST_MINUTES):
        current_minute = t_all[current_pos]
        target_minute = current_minute + FORECAST_MINUTES

        start_pos = current_pos - WINDOW_MINUTES + 1
        end_pos = current_pos + 1

        t_window = t_all[start_pos:end_pos]
        y_window = y_all[start_pos:end_pos]

        selected_degree, degree_info = choose_best_degree_short_window(
            t_window=t_window,
            y_window=y_window,
            candidate_degrees=CANDIDATE_DEGREES,
            validation_minutes=VALIDATION_MINUTES
        )

        pred_people_arr, effective_degree, coeffs = fit_poly_predict(
            x_train=t_window,
            y_train=y_window,
            x_target=[target_minute],
            degree=selected_degree
        )

        raw_pred_people = float(pred_people_arr[0])
        pred_people = stabilize_prediction(raw_pred_people, y_window)

        actual_people = float(pivot_df.loc[target_minute, area])

        pred_waiting = float(people_to_waiting_time(pred_people, area))
        actual_waiting = float(people_to_waiting_time(actual_people, area))

        pred_congestion = float(waiting_to_congestion(pred_waiting))
        actual_congestion = float(waiting_to_congestion(actual_waiting))

        results.append({
            "area": area,
            "current_minute": int(current_minute),
            "target_minute": int(target_minute),
            "window_start_minute": int(t_window[0]),
            "window_end_minute": int(t_window[-1]),
            "selected_degree": int(selected_degree),
            "effective_degree": int(effective_degree),

            "raw_pred_people": raw_pred_people,
            "pred_people": pred_people,
            "actual_people": actual_people,

            "last_people": float(y_window[-1]),
            "local_min_people": float(np.min(y_window)),
            "local_max_people": float(np.max(y_window)),

            "pred_waiting_time": pred_waiting,
            "actual_waiting_time": actual_waiting,

            "pred_congestion": pred_congestion,
            "actual_congestion": actual_congestion
        })

        degree_logs.append({
            "area": area,
            "current_minute": int(current_minute),
            "target_minute": int(target_minute),
            "selected_degree": int(selected_degree),
            "effective_degree": int(effective_degree),
            "degree_info": str(degree_info),
            "raw_pred_people": raw_pred_people,
            "stabilized_pred_people": pred_people
        })

pred_df = pd.DataFrame(results)
degree_log_df = pd.DataFrame(degree_logs)

# =========================================
# 7. 성능 평가
# =========================================
metrics = []

for area in pred_df["area"].unique():
    temp = pred_df[pred_df["area"] == area].sort_values("target_minute")

    metrics.append({
        "area": area,
        "n_samples": len(temp),

        "people_rmse": rmse(temp["actual_people"], temp["pred_people"]),
        "people_mae": mae(temp["actual_people"], temp["pred_people"]),

        "waiting_rmse": rmse(temp["actual_waiting_time"], temp["pred_waiting_time"]),
        "waiting_mae": mae(temp["actual_waiting_time"], temp["pred_waiting_time"]),

        "congestion_rmse": rmse(temp["actual_congestion"], temp["pred_congestion"]),
        "congestion_mae": mae(temp["actual_congestion"], temp["pred_congestion"]),
    })

metrics_df = pd.DataFrame(metrics).sort_values(
    ["congestion_rmse", "congestion_mae", "people_rmse"]
)

degree_stats_df = (
    pred_df.groupby(["area", "selected_degree"])
    .size()
    .reset_index(name="count")
    .sort_values(["area", "selected_degree"])
)

# =========================================
# 8. 저장
# =========================================
pred_df.to_csv(OUTPUT_PRED_CSV, index=False)
metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
degree_stats_df.to_csv(OUTPUT_DEGREE_CSV, index=False)
degree_log_df.to_csv(OUTPUT_DEGREE_LOG_CSV, index=False)

print(f"저장 완료: {OUTPUT_PRED_CSV}")
print(f"저장 완료: {OUTPUT_METRICS_CSV}")
print(f"저장 완료: {OUTPUT_DEGREE_CSV}")
print(f"저장 완료: {OUTPUT_DEGREE_LOG_CSV}")

print("\n=== 성능 상위 구역 ===")
print(metrics_df.head(10))

# =========================================
# 9. 구역별 그래프 저장
# =========================================
for area in pred_df["area"].unique():
    temp = pred_df[pred_df["area"] == area].sort_values("target_minute")

    if len(temp) == 0:
        continue

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # 사람 수
    axes[0].plot(
        temp["target_minute"], temp["actual_people"],
        label="Actual People", linewidth=2
    )
    axes[0].plot(
        temp["target_minute"], temp["pred_people"],
        label="Predicted People (Stable)", linewidth=2, linestyle="--"
    )
    axes[0].plot(
        temp["target_minute"], temp["raw_pred_people"],
        label="Raw Predicted People", linewidth=1, linestyle=":"
    )
    axes[0].set_title(f"{area} - Actual vs Predicted People (Stable 5min)")
    axes[0].set_ylabel("People Count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 대기시간
    axes[1].plot(
        temp["target_minute"], temp["actual_waiting_time"],
        label="Actual Waiting", linewidth=2
    )
    axes[1].plot(
        temp["target_minute"], temp["pred_waiting_time"],
        label="Predicted Waiting", linewidth=2, linestyle="--"
    )
    axes[1].set_title(f"{area} - Actual vs Predicted Waiting Time (Stable 5min)")
    axes[1].set_ylabel("Waiting Time (min)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 혼잡도
    axes[2].plot(
        temp["target_minute"], temp["actual_congestion"],
        label="Actual Congestion", linewidth=2
    )
    axes[2].plot(
        temp["target_minute"], temp["pred_congestion"],
        label="Predicted Congestion", linewidth=2, linestyle="--"
    )
    axes[2].set_title(f"{area} - Actual vs Predicted Congestion (Stable 5min)")
    axes[2].set_xlabel("Minute Index")
    axes[2].set_ylabel("Congestion")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"{area}_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

print(f"\n그래프 저장 완료: {GRAPH_DIR}")
