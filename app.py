import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


daily_df, movies_df = load_data()

# ---------------------------
# 기간 정보 (일별 표 기준)
# ---------------------------
date_col = "날짜"
min_date = str(daily_df[date_col].min())
max_date = str(daily_df[date_col].max())

st.subheader("📅 기준 기간")
st.write(f"박스오피스 일별 데이터 기준: **{min_date} ~ {max_date}**")

# ---------------------------
# 영화별 표 (원본) 표시
# ---------------------------
st.subheader("📄 영화별 표 (원본 상위 5행)")
st.dataframe(movies_df.head())

# ---------------------------
# 영화코드 순 정렬 및 train/test 분리
# ---------------------------
movies_sorted = movies_df.sort_values("movieCd").reset_index(drop=True)

# 열 편마다 앞의 세 편을 테스트용으로 (10편마다 처음 3편)
n = len(movies_sorted)
group_idx = np.arange(n) % 10
test_mask = group_idx < 3

test_df = movies_sorted[test_mask].copy()
train_df = movies_sorted[~test_mask].copy()

st.subheader("🧪 학습 / 평가 데이터 분리")
st.write(f"전체 영화 편수: **{n}편**")
st.write(f"학습에 사용한 영화 편수: **{len(train_df)}편**")
st.write(f"평가에 사용한 영화 편수: **{len(test_df)}편**")

# ---------------------------
# 변수 선택
# ---------------------------
st.subheader("🔧 사용할 변수 선택")

candidate_features = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부(1/0)",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 유지 일수",
}

selected_features = []
cols = st.columns(len(candidate_features))
for i, (feat, label) in enumerate(candidate_features.items()):
    with cols[i]:
        checked = st.checkbox(label, value=True, key=feat)
        if checked:
            selected_features.append(feat)

if len(selected_features) == 0:
    st.warning("최소 하나 이상의 변수를 선택해 주세요.")
    st.stop()

target = "total_audi"

# 결측치 제거 (선택된 변수 + target 기준)
use_cols = selected_features + [target]

train_clean = train_df.dropna(subset=use_cols)
test_clean = test_df.dropna(subset=use_cols)

if len(train_clean) == 0 or len(test_clean) == 0:
    st.error("결측치 제거 후 학습 또는 평가에 사용할 데이터가 없습니다.")
    st.stop()

X_train = train_clean[selected_features]
y_train = train_clean[target]
X_test = test_clean[selected_features]
y_test = test_clean[target]

# ---------------------------
# 모델 학습
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None)  # 음수 예측 방지

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 모델 성능 평가")
c1, c2, c3 = st.columns(3)
c1.metric("학습에 쓴 영화 편수", f"{len(X_train)}편")
c2.metric("평가에 쓴 영화 편수", f"{len(X_test)}편")
c3.metric("결정계수 (R²)", f"{r2:.3f}")

st.write(f"평균 절대 오차 (MAE): 약 **{mae:,.0f}명**")

# ---------------------------
# 산점도: 실제 vs 예측 (로그 스케일)
# ---------------------------
st.subheader("📈 실제 관객 수 vs 예측 관객 수 (로그 스케일)")

result_df = test_clean.copy()
result_df["predicted_total_audi"] = y_pred

# 1,000명 미만 예측은 바닥에 붙이기 위한 처리
FLOOR = 1000
below_floor_count = (result_df["predicted_total_audi"] < FLOOR).sum()

plot_df = result_df.copy()
plot_df["plot_pred"] = plot_df["predicted_total_audi"].clip(lower=FLOOR)
plot_df["plot_actual"] = plot_df["total_audi"].clip(lower=1)  # 로그축 0 방지

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df["plot_actual"],
    y=plot_df["plot_pred"],
    mode="markers",
    name="영화",
    text=plot_df["movieNm"] if "movieNm" in plot_df.columns else None,
    marker=dict(size=9, color="royalblue", opacity=0.7),
    hovertemplate="%{text}<br>실제: %{x:,.0f}<br>예측: %{y:,.0f}<extra></extra>"
))

min_val = min(plot_df["plot_actual"].min(), plot_df["plot_pred"].min())
max_val = max(plot_df["plot_actual"].max(), plot_df["plot_pred"].max())

fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    name="실제 = 예측",
    line=dict(color="red", dash="dash")
))

fig.update_xaxes(type="log", title="실제 총 관객 수 (로그)")
fig.update_yaxes(type="log", title="예측 총 관객 수 (로그)")
fig.update_layout(
    title="테스트 영화: 실제 vs 예측 총 관객 수",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.write(
    f"예측값이 1,000명보다 작게 나온 영화(그래프 바닥에 표시됨): "
    f"**{below_floor_count}편**"
)

# ---------------------------
# 결과 상세 표
# ---------------------------
st.subheader("🔍 테스트 영화별 예측 결과")
display_cols = ["movieCd"]
if "movieNm" in result_df.columns:
    display_cols.append("movieNm")
display_cols += [target, "predicted_total_audi"]

st.dataframe(
    result_df[display_cols].rename(columns={
        target: "실제 총 관객 수",
        "predicted_total_audi": "예측 총 관객 수"
    }).reset_index(drop=True)
)
