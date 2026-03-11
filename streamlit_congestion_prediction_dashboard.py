import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# 1. 기본 설정
# =========================================
st.set_page_config(
    page_title="구역별 혼잡도 예측 성능 대시보드",
    layout="wide"
)

st.title("구역별 혼잡도 예측 성능 대시보드")
st.markdown("5분 Rolling Polynomial Extrapolation + 튐 방지 안정화 결과")

REQUIRED_FILES = [
    "rolling_poly_5min_stable_prediction.csv",
    "rolling_poly_5min_stable_metrics.csv",
    "rolling_poly_5min_stable_selected_degree_stats.csv",
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missing_files:
    st.error("필수 결과 파일이 없습니다.")
    st.code("\n".join(missing_files))
    st.info("먼저 rolling_poly_5min_stable.py 를 실행해서 결과 csv 파일을 생성하세요.")
    st.stop()

# =========================================
# 2. 데이터 로드
# =========================================
@st.cache_data
def load_data():
    pred_df = pd.read_csv("rolling_poly_5min_stable_prediction.csv")
    metrics_df = pd.read_csv("rolling_poly_5min_stable_metrics.csv")
    degree_df = pd.read_csv("rolling_poly_5min_stable_selected_degree_stats.csv")

    pred_df["area"] = pred_df["area"].astype(str).str.strip()
    metrics_df["area"] = metrics_df["area"].astype(str).str.strip()
    degree_df["area"] = degree_df["area"].astype(str).str.strip()

    return pred_df, metrics_df, degree_df

pred_df, metrics_df, degree_df = load_data()

if metrics_df.empty:
    st.warning("metrics 데이터가 비어 있습니다.")
    st.stop()

# =========================================
# 3. 사이드바
# =========================================
st.sidebar.header("설정")

sort_column = st.sidebar.selectbox(
    "성능표 정렬 기준",
    options=[
        "congestion_rmse",
        "congestion_mae",
        "people_rmse",
        "people_mae",
        "waiting_rmse",
        "waiting_mae",
        "n_samples"
    ],
    index=0
)

ascending = st.sidebar.radio(
    "정렬 방식",
    options=["오름차순", "내림차순"],
    index=0
)

ascending_bool = (ascending == "오름차순")

sorted_metrics_df = metrics_df.sort_values(
    by=sort_column,
    ascending=ascending_bool
).reset_index(drop=True)

area_list = sorted_metrics_df["area"].tolist()
selected_area = st.sidebar.selectbox("구역 선택", area_list)

# =========================================
# 4. 상단 요약
# =========================================
st.subheader("전체 요약")

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 구역 수", len(metrics_df))
col2.metric("평균 혼잡도 RMSE", f"{metrics_df['congestion_rmse'].mean():.4f}")
col3.metric("평균 혼잡도 MAE", f"{metrics_df['congestion_mae'].mean():.4f}")
col4.metric("평균 사람 수 RMSE", f"{metrics_df['people_rmse'].mean():.4f}")

# =========================================
# 5. 성능표
# =========================================
st.subheader("구역별 예측 성능표")

st.dataframe(
    sorted_metrics_df,
    width="stretch",
    height=450
)

# =========================================
# 6. 선택 구역 상세 성능
# =========================================
st.subheader(f"선택 구역 상세 분석: {selected_area}")

area_metric = sorted_metrics_df[sorted_metrics_df["area"] == selected_area].iloc[0]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("People RMSE", f"{area_metric['people_rmse']:.4f}")
c2.metric("People MAE", f"{area_metric['people_mae']:.4f}")
c3.metric("Waiting RMSE", f"{area_metric['waiting_rmse']:.4f}")
c4.metric("Waiting MAE", f"{area_metric['waiting_mae']:.4f}")
c5.metric("Congestion RMSE", f"{area_metric['congestion_rmse']:.4f}")
c6.metric("Congestion MAE", f"{area_metric['congestion_mae']:.4f}")

# =========================================
# 7. 선택 구역 데이터
# =========================================
temp = pred_df[pred_df["area"] == selected_area].sort_values("target_minute").copy()

if temp.empty:
    st.warning("선택한 구역의 예측 결과가 없습니다.")
    st.stop()

# =========================================
# 8. 실제 vs 예측 그래프
# =========================================
st.subheader(f"{selected_area} 실제값 vs 예측값 비교")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# 사람 수
axes[0].plot(
    temp["target_minute"],
    temp["actual_people"],
    label="Actual People",
    linewidth=2
)
axes[0].plot(
    temp["target_minute"],
    temp["pred_people"],
    label="Predicted People (Stable)",
    linewidth=2,
    linestyle="--"
)
axes[0].plot(
    temp["target_minute"],
    temp["raw_pred_people"],
    label="Raw Predicted People",
    linewidth=1,
    linestyle=":"
)
axes[0].set_title(f"{selected_area} - Actual vs Predicted People")
axes[0].set_ylabel("People Count")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 대기시간
axes[1].plot(
    temp["target_minute"],
    temp["actual_waiting_time"],
    label="Actual Waiting Time",
    linewidth=2
)
axes[1].plot(
    temp["target_minute"],
    temp["pred_waiting_time"],
    label="Predicted Waiting Time",
    linewidth=2,
    linestyle="--"
)
axes[1].set_title(f"{selected_area} - Actual vs Predicted Waiting Time")
axes[1].set_ylabel("Waiting Time (min)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 혼잡도
axes[2].plot(
    temp["target_minute"],
    temp["actual_congestion"],
    label="Actual Congestion",
    linewidth=2
)
axes[2].plot(
    temp["target_minute"],
    temp["pred_congestion"],
    label="Predicted Congestion",
    linewidth=2,
    linestyle="--"
)
axes[2].set_title(f"{selected_area} - Actual vs Predicted Congestion")
axes[2].set_xlabel("Minute Index")
axes[2].set_ylabel("Congestion")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

# =========================================
# 9. 예측값 안정화 확인용 표
# =========================================
st.subheader(f"{selected_area} 예측 안정화 확인")

show_cols = [
    "target_minute",
    "last_people",
    "local_min_people",
    "local_max_people",
    "raw_pred_people",
    "pred_people",
    "actual_people",
    "pred_congestion",
    "actual_congestion",
    "selected_degree"
]

st.dataframe(
    temp[show_cols],
    width="stretch",
    height=300
)

# =========================================
# 10. 차수 선택 통계
# =========================================
st.subheader(f"{selected_area} 차수 선택 빈도")

degree_temp = degree_df[degree_df["area"] == selected_area].sort_values("selected_degree")

if len(degree_temp) > 0:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.dataframe(degree_temp, width="stretch")

    with col_b:
        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.bar(degree_temp["selected_degree"].astype(str), degree_temp["count"])
        ax.set_title(f"{selected_area} - Selected Degree Frequency")
        ax.set_xlabel("Selected Degree")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
else:
    st.info("해당 구역의 차수 선택 정보가 없습니다.")

# =========================================
# 11. 성능 상위 / 하위 구역
# =========================================
st.subheader("혼잡도 예측 성능 상위 / 하위 구역")

col_top, col_bottom = st.columns(2)

with col_top:
    st.markdown("#### 혼잡도 RMSE가 낮은 상위 5개 구역")
    top5 = metrics_df.sort_values("congestion_rmse", ascending=True).head(5)
    st.dataframe(top5, width="stretch")

with col_bottom:
    st.markdown("#### 혼잡도 RMSE가 높은 하위 5개 구역")
    bottom5 = metrics_df.sort_values("congestion_rmse", ascending=False).head(5)
    st.dataframe(bottom5, width="stretch")
