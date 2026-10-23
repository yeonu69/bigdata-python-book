# 04장. 그림으로 데이터와 대화하기: 탐색적 분석과 시각화 (개정판) — 04~05절 실습, 06절 실습, 미니 프로젝트 전체 코드
# '# ### 4-k' 조각 하나가 책의 [파이썬 실습 4-k] 하나입니다(4-1a, 4-1b처럼 단계로 나눈 것도 있습니다).
# M-k는 04절 손 계산을 파이썬으로 확인하는 짧은 코드, MP-k는 [미니 프로젝트 4]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: pandas, numpy, matplotlib, scikit-learn)
# 미니 프로젝트는 강의 자료 파일(서울_미세먼지_날씨_시간별_2023_2025.csv)이 같은 폴더에 있어야 합니다.

# ### 4-1a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib            # 코랩: 한글 글꼴을 자동으로 설정
except ImportError:                        # 나눔고딕 파일이 이미 있는 컴퓨터
    import glob, os
    import matplotlib.font_manager as fm
    for f in glob.glob(os.path.expanduser("~/.fonts/Nanum*.ttf")):
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False  # 음수 기호(-)가 깨지지 않게

URL = ("https://raw.githubusercontent.com/syedsharin/"
       "Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv")
df = pd.read_csv(URL, encoding="latin-1")          # °C 기호 때문에 latin-1
df.columns = ["date", "count", "hour", "temp", "humidity", "wind",
              "visibility", "dew", "solar", "rain", "snow", "season",
              "holiday", "functioning"]
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")   # 일/월/년

# 3장의 정제 규칙 세 가지를 그대로
h = df[df["functioning"] == "Yes"].copy()          # ① 운영한 시간만
h.loc[h["humidity"] == 0, "humidity"] = np.nan     # ② 습도 0%는 결측
hol = pd.to_datetime(["2017-12-25", "2018-01-01", "2018-02-15", "2018-02-16",
                      "2018-02-17", "2018-03-01", "2018-05-05", "2018-05-07",
                      "2018-05-22", "2018-06-06", "2018-06-13", "2018-08-15",
                      "2018-09-23", "2018-09-24", "2018-09-25", "2018-09-26",
                      "2018-10-03", "2018-10-09"])  # ③ 공식 공휴일 18일
h["dayoff"] = (h["date"].dt.dayofweek >= 5) | h["date"].isin(hol)  # 쉬는 날

day = (h.groupby("date")
         .agg(count=("count", "sum"), temp=("temp", "mean"),
              humidity=("humidity", "mean"), rain=("rain", "sum"),
              hours=("hour", "size"), dayoff=("dayoff", "first"))
         .reset_index())
day = day[day["hours"] == 24].reset_index(drop=True)  # 24시간이 다 있는 날
print("운영한 시간:", len(h), "/ 하루 단위 표:", len(day), "일")
print("평일", (~day["dayoff"]).sum(), "일, 쉬는 날", day["dayoff"].sum(), "일")

# ### 4-1b
fig, ax = plt.subplots(figsize=(9, 3.5))           # 도화지 1장, 칸 1개
ax.plot(day["date"], day["count"], color="lightgray", label="하루 대여량")
week = day["count"].rolling(7, center=True).mean() # 앞뒤 7일 평균
ax.plot(day["date"], week, color="tab:red", label="7일 이동평균")
ax.set_ylabel("하루 대여량(대)")
ax.legend()
plt.show()

month = day.groupby(day["date"].dt.month)["count"].mean()
print("1월 하루 평균:", round(month[1]), "대 / 6월:", round(month[6]), "대")
print(day.nlargest(3, "count")[["date", "count", "rain"]].to_string(index=False))
print(day.nsmallest(3, "count")[["date", "count", "rain"]].to_string(index=False))

# ### 4-2
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))  # 칸 2개를 가로로
axes[0].hist(h["count"], bins=30, color="tab:blue", edgecolor="white")
axes[0].set_title("보통 눈금")
log_bins = np.logspace(np.log10(2), np.log10(3556), 31)  # 같은 배수로 늘어나는 경계
axes[1].hist(h["count"], bins=log_bins, color="tab:blue", edgecolor="white")
axes[1].set_xscale("log")                          # 가로축을 로그 눈금으로
axes[1].set_title("로그 눈금")
for ax in axes:
    ax.set_xlabel("한 시간 대여량(대)")
axes[0].set_ylabel("시간 수")
plt.show()

print(f"평균 {h['count'].mean():.1f}대, 중앙값 {h['count'].median():.0f}대")
small = h[h["count"] < 100]                        # 100대 미만인 시간
print(f"100대 미만: {len(small)}시간 ({len(small) / len(h):.1%})")
print(f"그 가운데 새벽 3~6시: {small['hour'].between(3, 6).mean():.1%}")
print(f"비 온 시간의 비율: 100대 미만에서 {(small['rain'] > 0).mean():.1%},",
      f"전체에서 {(h['rain'] > 0).mean():.1%}")

# ### 4-3
day["month"] = day["date"].dt.month
fig, ax = plt.subplots(figsize=(9, 3.6))
day.boxplot(column="count", by="month", ax=ax, grid=False)
ax.set_title("")
fig.suptitle("")                                   # 판다스가 붙이는 제목 지우기
ax.set_xlabel("월")
ax.set_ylabel("하루 대여량(대)")
plt.show()

q1, q3 = day["count"].quantile([0.25, 0.75])       # 아래·위 사분위수
print("1년 전체: Q1", round(q1), "/ 중앙값", round(day["count"].median()),
      "/ Q3", round(q3), "/ IQR", round(q3 - q1))
iqr = day.groupby("month")["count"].quantile(0.75) - day.groupby("month")["count"].quantile(0.25)
print("상자가 가장 긴 달:", iqr.idxmax(), "월, IQR", round(iqr.max()))

def far(g):                                         # 수염 밖으로 떨어진 날
    a, b = g["count"].quantile([0.25, 0.75])
    return g[(g["count"] < a - 1.5 * (b - a)) | (g["count"] > b + 1.5 * (b - a))]
odd = day.groupby("month", group_keys=False).apply(far)
print("수염 밖의 날:", len(odd), "일, 그 가운데 비가 5mm 넘게 온 날:", (odd["rain"] > 5).sum(), "일")

# ### 4-4
prof = h.groupby(["hour", "dayoff"])["count"].mean().unstack()
prof.columns = ["평일", "쉬는 날"]
ax = prof.plot(figsize=(9, 3.6), marker="o", ms=4, color=["tab:blue", "tab:red"])
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("시각(시)")
ax.set_ylabel("시간당 평균 대여량(대)")
plt.show()

mixed = h.groupby("hour")["count"].mean()          # 평일과 쉬는 날을 섞은 평균
print("섞은 평균:", mixed[[4, 8, 18]].round().astype(int).to_dict())
print(prof.loc[[4, 8, 12, 17, 18]].round().astype(int))

# ### 4-5
h["month"] = h["date"].dt.month
table = h.pivot_table(index="hour", columns="month",
                      values="count", aggfunc="mean")   # 24줄 × 12칸
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(table, cmap="Reds", aspect="auto")  # 숫자를 색으로
ax.set_xticks(range(12), table.columns)
ax.set_yticks(range(0, 24, 2), range(0, 24, 2))
ax.set_xlabel("월")
ax.set_ylabel("시각(시)")
fig.colorbar(im, label="시간당 평균 대여량(대)")
plt.show()

print("18시 줄:", table.loc[18].round().astype(int).to_dict())
t18 = h[h["hour"] == 18].groupby("month")["temp"].mean()
print("18시 평균기온:", t18[[6, 7, 8, 9]].round(1).to_dict())
print("달마다 강수량 합:", h.groupby("month")["rain"].sum()[[5, 6, 7, 8, 9]].round(1).to_dict())

# ### 4-6a
bands = pd.cut(day["temp"], bins=range(-16, 40, 4))   # 4도씩: -16도 초과 ~ 36도 이하
print("구간에 들지 못한 날:", bands.isna().sum(), "일")
by = day.groupby(bands, observed=True).agg(날수=("count", "size"),
                                            평균기온=("temp", "mean"),
                                            평균대여=("count", "mean"))
print(by.tail(5).round(1).to_string())
hot = day["temp"] > 28                             # 28도 넘는 날을 한데 모으면
print("28도 넘는 날:", hot.sum(), "일, 하루 평균", round(day.loc[hot, "count"].mean()), "대")

rainy = day["rain"] >= 1                           # 하루 1mm 이상 비 온 날
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
axes[0].scatter(day["temp"], day["count"], s=10, alpha=0.5, color="tab:blue")
axes[0].plot(by["평균기온"], by["평균대여"], color="tab:red", marker="o")
axes[0].set_xlabel("하루 평균기온(°C)")
axes[0].set_ylabel("하루 대여량(대)")
axes[1].scatter(day.loc[~rainy, "humidity"], day.loc[~rainy, "count"],
                s=10, alpha=0.5, color="tab:blue", label="비 안 온 날")
axes[1].scatter(day.loc[rainy, "humidity"], day.loc[rainy, "count"],
                s=12, alpha=0.8, color="tab:red", label="비 온 날")
axes[1].set_xlabel("하루 평균습도(%)")
axes[1].legend()
plt.show()

# ### 4-6b
r_temp = day["count"].corr(day["temp"])
r_hum = day["count"].corr(day["humidity"])
r_rain = day["count"].corr(day["rain"])
print(f"상관계수  기온 {r_temp:.2f}  습도 {r_hum:.2f}  강수량 {r_rain:.2f}")
wet = day[day["humidity"] > 80]
print("습도 80% 넘는 날:", len(wet), "일, 그 가운데 비 온 날:", (wet["rain"] >= 1).sum(), "일")
print("습도 40% 이하인 날의 평균기온:", round(day.loc[day["humidity"] <= 40, "temp"].mean(), 1), "도")

# ### 4-7
means = day.groupby("dayoff")["count"].mean()
means.index = ["평일", "쉬는 날"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].bar(means.index, means.values, color=["tab:blue", "tab:red"])
axes[0].set_title("정직한 축 (0부터)")
axes[1].bar(means.index, means.values, color=["tab:blue", "tab:red"])
axes[1].set_ylim(16000, 18500)                     # 세로축을 16,000부터
axes[1].set_title("잘린 축 (16,000부터)")
plt.show()

real = means["평일"] / means["쉬는 날"]
shown = (means["평일"] - 16000) / (means["쉬는 날"] - 16000)
print(means.round().astype(int).to_dict())
print(f"실제 비 {real:.3f}, 잘린 축의 막대 높이 비 {shown:.1f}")
print(f"거짓말 계수 {(shown - 1) / (real - 1):.0f}")
weekend = day["date"].dt.dayofweek >= 5            # 토·일만 주말로 (공휴일은 평일 쪽)
by_we = day.groupby(weekend)["count"].agg(["size", "mean"]).round().astype(int)
by_we.index = ["평일(공휴일 포함)", "주말(토·일)"]
print(by_we.to_string())
print("차이:", by_we["mean"].iloc[0] - by_we["mean"].iloc[1], "대")

# ### 4-8
seasons = ["Winter", "Spring", "Summer", "Autumn"]
names = ["겨울", "봄", "여름", "가을"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)  # 축을 함께
for ax, s, name in zip(axes.flat, seasons, names):
    part = h[h["season"] == s]
    p = part.groupby(["hour", "dayoff"])["count"].mean().unstack()
    ax.plot(p.index, p[False], color="tab:blue", label="평일")
    ax.plot(p.index, p[True], color="tab:red", label="쉬는 날")
    ax.set_title(name)
    ax.set_xticks(range(0, 24, 6))
axes[1, 0].set_xlabel("시각(시)")
axes[1, 0].set_ylabel("시간당 평균 대여량(대)")
axes[0, 0].legend()
plt.show()
summer = h[h["season"] == "Summer"]
print("여름 쉬는 날 가장 많은 시각:", summer[summer["dayoff"]].groupby("hour")["count"].mean().idxmax(), "시")

# ### 4-9
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ans = {"I": (x, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
       "II": (x, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
       "III": (x, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
       "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
              [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True)
for ax, (name, (xs, ys)) in zip(axes, ans.items()):
    xs, ys = np.array(xs), np.array(ys)
    b1, b0 = np.polyfit(xs, ys, 1)                 # 가장 잘 맞는 직선
    r = np.corrcoef(xs, ys)[0, 1]
    print(f"{name:3s} x평균 {xs.mean():.1f}  y평균 {ys.mean():.2f}  y분산 {ys.var(ddof=1):.2f}"
          f"  상관 {r:.3f}  직선 y = {b0:.2f} + {b1:.3f}x")
    ax.scatter(xs, ys, color="tab:blue")
    ax.plot([3, 20], [b0 + b1 * 3, b0 + b1 * 20], color="tab:red")
    ax.set_title(name)
plt.show()

# ### 4-10
months_kr = ["4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월",
             "12월", "1월", "2월", "3월"]
disease = [1.4, 6.2, 4.7, 150, 328.5, 312.2, 197, 340.6, 631.5, 1022.8, 822.8, 480.3,
           177.5, 171.8, 247.6, 107.5, 129.9, 47.5, 32.8, 56.4, 25.3, 11.4, 6.6, 3.9]
wounds = [0, 0, 0, 0, 0.4, 32.1, 51.7, 115.8, 41.7, 30.7, 16.3, 12.8,
          17.9, 16.6, 64.5, 37.7, 44.1, 69.4, 13.6, 10.5, 5, 0.5, 0, 0]
other = [7, 4.6, 2.5, 9.6, 11.9, 27.7, 50.1, 42.8, 48, 120, 140.1, 68.6,
         21.2, 12.5, 9.6, 9.3, 6.7, 5, 4.6, 10.1, 7.8, 13, 5.2, 9.1]
theta = np.arange(12) * 2 * np.pi / 12             # 한 달 = 30도
fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), subplot_kw={"projection": "polar"})
titles = ["1854년 4월 ~ 1855년 3월", "1855년 4월 ~ 1856년 3월"]
for k, ax in enumerate(axes):
    part = slice(12 * k, 12 * k + 12)
    for vals, color, name in [(disease, "#C8234B", "막을 수 있던 병"),
                              (other, "#555555", "그 밖의 원인"),
                              (wounds, "#7FB3D5", "전투 부상")]:
        ax.bar(theta, np.sqrt(vals[part]), width=2 * np.pi / 12,   # 넓이가 값에 비례하게
               color=color, alpha=0.8, edgecolor="white", label=name)
    ax.set_theta_zero_location("N")                # 4월을 12시 방향에
    ax.set_theta_direction(-1)                     # 시계 방향으로
    ax.set_xticks(theta, months_kr)
    ax.set_yticklabels([])
    ax.set_ylim(0, np.sqrt(1022.8))                # 두 그림의 크기를 같게
    ax.set_title(titles[k], pad=18)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3)
plt.show()
n_disease, n_wounds, n_other = 14476, 1758, 1748   # 24개월 사망자 수 합계
total = n_disease + n_wounds + n_other
print(f"병 {n_disease:,}명 ({n_disease / total:.1%}), 부상 {n_wounds:,}명, "
      f"그 밖 {n_other:,}명 / 병은 부상의 {n_disease / n_wounds:.1f}배")

# ### 4-11
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape)                                     # (장 수, 칸 수)
maps = {"PCA": PCA(n_components=2).fit_transform(X),
        "t-SNE": TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (name, xy) in zip(axes, maps.items()):
    ax.scatter(xy[:, 0], xy[:, 1], c=y, cmap="tab10", s=6)
    for d in range(10):                            # 숫자마다 가운데에 이름표
        cx, cy = np.median(xy[y == d], axis=0)
        ax.text(cx, cy, str(d), fontsize=14, weight="bold", ha="center", va="center")
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1]   # 자기 다음으로 가까운 점
    agree = (y[idx] == y).mean()
    ax.set_title(f"{name}: 이웃이 같은 숫자 {agree:.1%}")
    ax.set_xticks([]); ax.set_yticks([])
    print(name, round(agree, 3))
plt.show()

# ### 4-12
naive = df.groupby("hour")["count"].mean()         # 운영 여부를 거르지 않음
clean = h.groupby("hour")["count"].mean()          # 운영한 시간만
print("오후 6시 평균:", round(naive[18], 1), "→", round(clean[18], 1))
daily_all = df.groupby("date")["count"].sum()      # 운영 안 한 날도 하루로 셈
oct_all = daily_all[daily_all.index.month == 10]
oct_clean = day.loc[day["date"].dt.month == 10, "count"]
print(f"10월 하루 평균: {oct_all.mean():,.0f}대 ({len(oct_all)}일) → "
      f"{oct_clean.mean():,.0f}대 ({len(oct_clean)}일)")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(oct_all.index.day, oct_all.values, color="tab:blue")
ax.set_xlabel("2018년 10월(일)")
ax.set_ylabel("하루 대여량(대)")
plt.show()
print("10월에 0대인 날:", list(oct_all[oct_all == 0].index.day))

# ### M-1
v = pd.Series([17730, 22593, 24404, 6852, 977, 20144, 26285, 25670, 25861])
q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
iqr = q3 - q1
print(sorted(v))
print("Q1", q1, "중앙값", q2, "Q3", q3, "IQR", iqr)
print("1.5 × IQR =", 1.5 * iqr, "/ 아래 울타리", q1 - 1.5 * iqr, "→ 바깥 값", list(v[v < q1 - 1.5 * iqr]))

# ### M-2
t = np.array([4, 17, 18, 29, 32])                  # 다섯 날의 기온(도)
k = np.array([7, 19, 28, 27, 24])                  # 대여량(천 대)
zt = (t - t.mean()) / t.std(ddof=1)                # (값 - 평균) ÷ 표준편차
zk = (k - k.mean()) / k.std(ddof=1)
print("평균을 뺀 기온:", t - t.mean(), "제곱합:", ((t - t.mean()) ** 2).sum())
print("표준편차:", round(t.std(ddof=1), 2), round(k.std(ddof=1), 2))
print("기온 z:", zt.round(2))
print("대여 z:", zk.round(2))

# ### M-3
prod = zt * zk                                     # 날마다 z끼리 곱하기
print("곱:", prod.round(2), "합:", round(prod.sum(), 2))
r = prod.sum() / (len(t) - 1)                      # 더해서 n-1로 나눈다
print("손 계산 방식:", round(r, 3), "/ np.corrcoef:", round(np.corrcoef(t, k)[0, 1], 3))

# ### M-4
print(np.log10([119, 1755]).round(3), "차이", round(np.log10(1755 / 119), 3),
      "→", round(1755 / 119, 1), "배")

# ### MP-0
# ☆ 에어코리아·기상청에서 받은 원자료로 강의 자료 파일 만들기 (강의 자료의 파일을 쓰면 건너뜁니다)
import glob
AIR = sorted(glob.glob("20??년 *월.xlsx"))         # 에어코리아 확정자료, 한 달에 한 파일
if not AIR:
    print("받은 원자료가 없습니다. 강의 자료의 파일로 1단계를 하십시오.")
else:
    parts = []
    for f in AIR:                                  # 파일 하나에 1분쯤 걸립니다
        t = pd.read_excel(f)
        t = t[t["지역"].str.startswith("서울") & (t["망"] == "도시대기")]
        parts.append(t[["측정소명", "측정일시", "PM10", "PM25"]])
    air = pd.concat(parts, ignore_index=True)
    mp = (air.groupby("측정일시")
             .agg(PM10=("PM10", "mean"), PM25=("PM25", "mean"),
                  측정소수=("PM25", "count")).round(1).reset_index())
    # 측정일시의 시는 01~24: 24시를 다음 날 0시로 바꾼다
    d = pd.to_datetime((mp["측정일시"] // 100).astype(str), format="%Y%m%d")
    mp["일시"] = d + pd.to_timedelta(mp["측정일시"] % 100, unit="h")
    print(len(AIR), "개 파일 →", mp.shape, mp["일시"].min(), "~", mp["일시"].max())

# ### MP-1
PATH = "서울_미세먼지_날씨_시간별_2023_2025.csv"   # 코랩 왼쪽 파일 창에 올린 파일
air = pd.read_csv(PATH)
print(air.shape)
print(air.head(3).to_string())
print(air.isna().sum().to_dict())                  # 열마다 빈칸 수

try:
    pd.to_datetime(air["측정일시"].astype(str), format="%Y%m%d%H")
except ValueError as e:
    print("날짜로 바꾸지 못함:", str(e)[:58])
print("측정일시의 시:", (air["측정일시"] % 100).min(), "~", (air["측정일시"] % 100).max())
air["일시"] = pd.to_datetime(air["일시"])          # 24시를 다음 날 0시로 바꿔 둔 열
print("측정소가 25곳보다 적은 시각:", (air["측정소수"] < 25).sum(),
      "/ 가장 적을 때", air["측정소수"].min(), "곳")
print("비가 온 시각:", (air["강수량"] > 0).sum(), "/ 강수량이 빈칸인 시각:", air["강수량"].isna().sum())

# ### MP-2
print(air[["PM10", "PM25"]].describe().round(1).to_string())
print(f"PM2.5 '나쁨'(36 이상) 시각의 비율: {(air['PM25'] >= 36).mean():.1%}")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(air["PM25"], bins=np.logspace(0, np.log10(140), 31), color="tab:blue", edgecolor="white")
ax.set_xscale("log")
ax.axvline(36, color="tab:red", linestyle="--")    # '나쁨'이 시작되는 값
ax.set_xlabel("PM2.5 농도(로그 눈금)")
ax.set_ylabel("시각 수")
plt.show()

# ### MP-3
air["시"] = air["일시"].dt.hour
air["월"] = air["일시"].dt.month
air["요일"] = air["일시"].dt.dayofweek               # 0=월 … 6=일
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
air.groupby("시")["PM25"].mean().plot(ax=axes[0], marker="o", color="tab:blue")
air.groupby("월")["PM25"].mean().plot(ax=axes[1], marker="o", color="tab:red")
air.groupby("요일")["PM25"].mean().plot.bar(ax=axes[2], color="tab:gray")
axes[0].set_ylabel("PM2.5 평균 농도")
axes[1].set_xticks(range(1, 13))
axes[2].set_xticks(range(7), list("월화수목금토일"), rotation=0)
for ax, name in zip(axes, ["시각", "월", "요일"]):
    ax.set_xlabel(name)
plt.show()
by_h = air.groupby("시")["PM25"].mean()
by_m = air.groupby("월")["PM25"].mean()
print("시각별 평균:", round(by_h.min(), 1), "~", round(by_h.max(), 1))
print("가장 높은 달:", by_m.idxmax(), "월", round(by_m.max(), 1), "/ 가장 낮은 달:", by_m.idxmin(), "월", round(by_m.min(), 1))

# ### MP-4
air["날짜"] = air["일시"].dt.normalize()
daily = air.groupby("날짜").agg(PM10=("PM10", "mean"), PM25=("PM25", "mean"), 황사=("황사", "max"))
print(daily.nlargest(5, "PM10").round(1).to_string())
print("황사가 관측된 날:", daily["황사"].sum(), "일 /", air["황사"].sum(), "시간")
top = air.loc[air["PM10"].idxmax()]
print("PM10이 가장 높은 시각:", top["일시"], top["PM10"])
ratio = air["PM25"] / air["PM10"]                  # 미세먼지 가운데 초미세먼지의 몫
print("PM2.5 ÷ PM10 중앙값  황사 시각", round(ratio[air["황사"] == 1].median(), 2),
      "/ 그 밖의 시각", round(ratio[air["황사"] == 0].median(), 2))
dust = air["황사"] == 1
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter(air.loc[~dust, "PM10"], air.loc[~dust, "PM25"], s=3, alpha=0.3, color="tab:blue", label="그 밖의 시각")
ax.scatter(air.loc[dust, "PM10"], air.loc[dust, "PM25"], s=8, alpha=0.8, color="tab:red", label="황사 관측")
ax.set_xlabel("PM10 농도")
ax.set_ylabel("PM2.5 농도")
ax.legend()
plt.show()

# ### EX-5
off = h[h["dayoff"]]
t_off = off.pivot_table(index="hour", columns="month", values="count", aggfunc="mean")
print("쉬는 날, 달마다 가장 많이 빌린 시각:", t_off.idxmax().to_dict())

# ### EX-6
rain_h = air[air["강수량"] > 0]["PM25"].mean()
dry_h = air[air["강수량"].isna()]["PM25"].mean()
print("비 온 시각", round(rain_h, 1), "/ 강수량 빈칸 시각", round(dry_h, 1))
