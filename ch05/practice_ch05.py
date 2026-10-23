# 05장. 우연인가 아닌가: 통계적 추론 (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 5-k' 조각 하나가 책의 [파이썬 실습 5-k] 하나입니다. ★ 조각은 앞의 ★ 조각만 있으면 돌아갑니다.
# MP-k는 [미니 프로젝트 5]의 단계, EQ-k는 04절의 "같은 계산을 파이썬으로", EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_rev.py  (필요 라이브러리: pandas, numpy, scipy, matplotlib, holidays)

# ### 5-1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
try:
    import koreanize_matplotlib            # 코랩: 한글 글꼴을 자동으로 설정
except ImportError:                        # 나눔고딕 파일이 이미 있는 컴퓨터
    import glob, os
    import matplotlib.font_manager as fm
    for f in glob.glob(os.path.expanduser("~/.fonts/Nanum*.ttf")):
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

URL = ("https://raw.githubusercontent.com/syedsharin/"
       "Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv")
df = pd.read_csv(URL, encoding="latin-1")
df.columns = ["date", "count", "hour", "temp", "humidity", "wind", "visibility",
              "dew", "solar", "rain", "snow", "season", "holiday", "functioning"]
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
print(df.shape)

off = df[df["functioning"] == "No"]                       # 운영하지 않은 시간
print("운영하지 않은 날:", off["date"].nunique(), "일,", len(off), "시간, 대여량 합", off["count"].sum())
h = df[df["functioning"] == "Yes"]
day = (h.groupby("date")
         .agg(count=("count", "sum"), rain=("rain", "sum"), temp=("temp", "mean"),
              season=("season", "first"), hours=("hour", "size"))
         .reset_index())
day = day[day["hours"] == 24]                             # 24시간이 다 있는 날만
day["rainy"] = day["rain"] >= 1.0                         # 하루 1mm 이상이면 비 온 날
day["weekend"] = day["date"].dt.dayofweek >= 5            # 토요일(5), 일요일(6)
print(len(day), "일")
print(day.groupby("rainy")["count"].agg(["size", "mean", "std"]).round(0))

whole = df.groupby("date").agg(count=("count", "sum"), season=("season", "first"))
print("가을 하루 평균  0을 그대로 두면:", round(whole.loc[whole["season"] == "Autumn", "count"].mean()),
      "/ 운영한 날만:", round(day.loc[day["season"] == "Autumn", "count"].mean()))

rng = np.random.default_rng(0)                            # 점이 겹치지 않게 옆으로 흩뜨리기
for k, (name, g) in enumerate(day.groupby("rainy")):
    plt.scatter(k + rng.uniform(-0.25, 0.25, len(g)), g["count"], s=10, color="#9a9a9a")
    plt.hlines(g["count"].mean(), k - 0.3, k + 0.3, color="#C8234B", linewidth=3)
plt.xticks([0, 1], ["맑은 날", "비 온 날"])
plt.ylabel("하루 대여량 (대)")
plt.show()

# ### 5-2
rng = np.random.default_rng(42)
heads = rng.binomial(n=10, p=0.5, size=100_000)          # 동전 열 번 던지기를 10만 세트
print("10번 중 6번 이상 앞면:", (heads >= 6).mean())
print("이항검정 p값(10번 중 6번):", round(stats.binomtest(6, 10, 0.5).pvalue, 3))
print("이항검정 p값(100번 중 60번):", round(stats.binomtest(60, 100, 0.5).pvalue, 3))
from math import comb
print("홍차 8잔을 우연히 모두 맞힐 확률: 1 /", comb(8, 4), "=", round(1 / comb(8, 4), 3))

counts = np.bincount(heads, minlength=11)
plt.bar(range(11), counts, color=["#C8234B" if k >= 6 or k <= 4 else "#9a9a9a" for k in range(11)])
plt.xlabel("10번 중 앞면 수")
plt.ylabel("세트 수")
plt.show()

# ### 5-3
rainy = day.loc[day["rainy"], "count"]
dry = day.loc[~day["rainy"], "count"]
obs = dry.mean() - rainy.mean()                           # 실제 차이
rng = np.random.default_rng(42)
values = day["count"].to_numpy()
n_rainy = len(rainy)
diffs = np.empty(10_000)
for i in range(10_000):
    shuffled = rng.permutation(values)                    # 이름표를 섞는다
    diffs[i] = shuffled[n_rainy:].mean() - shuffled[:n_rainy].mean()
print(f"실제 차이 {obs:,.0f}대")
print(f"섞었을 때 차이의 표준편차 {diffs.std():,.0f}대")
print("실제만큼 벌어진 경우:", (np.abs(diffs) >= abs(obs)).sum(), "번 / 1만 번")

plt.hist(diffs, bins=50, color="#9a9a9a")
plt.axvline(obs, color="#C8234B", linewidth=3)
plt.xlabel("섞어서 만든 평균 차이 (대)")
plt.ylabel("횟수")
plt.show()

# ### 5-4
res = stats.ttest_ind(dry, rainy, equal_var=False)       # 웰치 t검정
ci = res.confidence_interval(0.95)
print(f"t = {res.statistic:.2f}, 자유도 {res.df:.0f}, p = {res.pvalue:.5f}")
print(f"차이의 95% 신뢰구간 {ci.low:,.0f} ~ {ci.high:,.0f}대")

x = day["count"]
m, se = x.mean(), x.std() / np.sqrt(len(x))               # 표준오차 = s / √n
lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se)
print(f"하루 평균 {m:,.0f}대, 표준오차 {se:,.0f}대, 95% 신뢰구간 {lo:,.0f} ~ {hi:,.0f}대")

n1, n2 = len(dry), len(rainy)
sp = np.sqrt(((n1 - 1) * dry.var() + (n2 - 1) * rainy.var()) / (n1 + n2 - 2))  # 합동 표준편차
print(f"효과 크기 d = {obs / sp:.2f}, 맑은 날보다 {obs / dry.mean() * 100:.0f}% 적음")

plt.figure(figsize=(7, 2.2))
plt.errorbar([obs], [0], xerr=[[obs - ci.low], [ci.high - obs]], fmt="o", color="#C8234B",
             capsize=8, linewidth=3, markersize=10)
plt.axvline(0, color="#9a9a9a", linestyle="--")
plt.yticks([])
plt.xlabel("맑은 날 - 비 온 날, 하루 대여량 차이 (대)")
plt.xlim(-1000, 8000)
plt.show()

# ### 5-5
true_mean = day["count"].mean()                           # 352일을 모집단으로 삼는다
rng = np.random.default_rng(7)
hit, lines = 0, []
for i in range(100):
    s = rng.choice(day["count"].to_numpy(), 30, replace=False)   # 30일 뽑기
    lo, hi = stats.t.interval(0.95, 29, loc=s.mean(), scale=s.std(ddof=1) / np.sqrt(30))
    hit += lo <= true_mean <= hi
    lines.append((lo, hi))
print("진짜 평균을 품은 신뢰구간:", hit, "개 / 100개")
boot = [rng.choice(dry, len(dry)).mean() - rng.choice(rainy, len(rainy)).mean()
        for _ in range(10_000)]                           # 부트스트랩
print("부트스트랩 95% 신뢰구간:", np.percentile(boot, [2.5, 97.5]).round(0))

for i, (lo, hi) in enumerate(lines):
    miss = not (lo <= true_mean <= hi)
    plt.vlines(i, lo, hi, color="#C8234B" if miss else "#9a9a9a", linewidth=3 if miss else 1.5)
plt.axhline(true_mean, color="black", linestyle="--")
plt.xlabel("신뢰구간 번호")
plt.ylabel("하루 평균 대여량 (대)")
plt.show()

# ### 5-6
band = day[(day["temp"] > 0) & (day["temp"] <= 10)]        # 하루 평균기온 0~10도인 날
print(band.groupby(band["date"].dt.to_period("M"))["count"].agg(["size", "mean"]).round(0))

rows = []
for s, name in [("Winter", "겨울"), ("Spring", "봄"), ("Summer", "여름"), ("Autumn", "가을")]:
    part = day[day["season"] == s]
    a = part.loc[~part["rainy"], "count"]
    b = part.loc[part["rainy"], "count"]
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    rows.append([name, len(a), round(a.mean()), len(b), round(b.mean()), round(a.mean() - b.mean()), round(p, 6)])
season = pd.DataFrame(rows, columns=["계절", "맑은 날", "평균", "비 온 날", "평균 ", "차이", "p값"])
print(season.to_string(index=False))

season.set_index("계절")[["평균", "평균 "]].plot.bar(color=["#9a9a9a", "#C8234B"], rot=0)
plt.legend(["맑은 날", "비 온 날"])
plt.ylabel("하루 평균 대여량 (대)")
plt.xlabel("")
plt.show()

# ### 5-7
wd = day.loc[~day["weekend"], "count"]
we = day.loc[day["weekend"], "count"]
res = stats.ttest_ind(wd, we, equal_var=False)
ci = res.confidence_interval(0.95)
print(f"평일 {len(wd)}일 {wd.mean():,.0f}대, 주말 {len(we)}일 {we.mean():,.0f}대")
print(f"차이 {wd.mean() - we.mean():,.0f}대, p = {res.pvalue:.3f}, 95% 신뢰구간 {ci.low:,.0f} ~ {ci.high:,.0f}대")

d = wd.mean() - we.mean()
plt.figure(figsize=(7, 2.2))
plt.errorbar([d], [0], xerr=[[d - ci.low], [ci.high - d]], fmt="o", color="#C8234B",
             capsize=8, linewidth=3, markersize=10)
plt.axvline(0, color="#9a9a9a", linestyle="--")
plt.yticks([])
plt.xlabel("평일 - 주말, 하루 대여량 차이 (대)")
plt.xlim(-1000, 8000)
plt.show()

# ### 5-8
rng = np.random.default_rng(2018)
ps = []
for _ in range(20):
    label = rng.random(len(day)) < 0.5                    # 동전으로 두 편 나누기
    ps.append(stats.ttest_ind(day["count"][label], day["count"][~label], equal_var=False).pvalue)
ps = np.array(ps)
print("20번 가운데 p < 0.05:", (ps < 0.05).sum(), "번")
count = 0
for _ in range(1000):
    label = rng.random(len(day)) < 0.5
    count += stats.ttest_ind(day["count"][label], day["count"][~label], equal_var=False).pvalue < 0.05
print("1,000번 가운데 p < 0.05:", count, "번")

plt.scatter(range(1, 21), ps, color=["#C8234B" if p < 0.05 else "#9a9a9a" for p in ps], s=60)
plt.axhline(0.05, color="#C8234B", linestyle="--")
plt.yscale("log")
plt.yticks([0.01, 0.05, 0.1, 0.5, 1], ["0.01", "0.05", "0.1", "0.5", "1"])
plt.xticks(range(1, 21))
plt.xlabel("의미 없이 나눈 검정 (번째)")
plt.ylabel("p값 (로그 눈금)")
plt.show()

# ### 5-9
pop = h["count"].to_numpy()                                    # 운영한 8,465시간 전체
big = h.loc[h["season"].isin(["Summer", "Autumn"]), "count"]   # 여름·가을만 모은 큰 표본
rng = np.random.default_rng(5)
small = rng.choice(pop, 100, replace=False)                    # 무작위 100시간
lo, hi = stats.t.interval(0.95, 99, loc=small.mean(), scale=small.std(ddof=1) / 10)
print(f"진짜 평균 {pop.mean():.1f}대")
print(f"큰 표본 {len(big):,}시간: {big.mean():.1f}대")
print(f"무작위 100시간: {small.mean():.1f}대 (95% 신뢰구간 {lo:.1f} ~ {hi:.1f})")
cover = 0
for _ in range(1000):
    s = rng.choice(pop, 100, replace=False)
    lo, hi = stats.t.interval(0.95, 99, loc=s.mean(), scale=s.std(ddof=1) / 10)
    cover += lo <= pop.mean() <= hi
print("무작위 100시간 신뢰구간 1,000개 가운데 진짜 평균을 품은 것:", cover, "개")

# ### 5-10
hh = h.assign(busy=h["count"] > h["count"].median(), raining=h["rain"] > 0)
table = pd.crosstab(hh["raining"], hh["busy"])
print(table)
chi = stats.chi2_contingency(table)
print(f"카이제곱 = {chi.statistic:.1f}, p = {chi.pvalue:.1e}")
print(hh.groupby(["season", "raining"])["busy"].mean().unstack().round(3))

# ### 5-11
r = stats.pearsonr(day["temp"], day["count"])
print(f"기온-대여량 r = {r.statistic:.3f}, 95% 신뢰구간 {r.confidence_interval().low:.3f} ~ {r.confidence_interval().high:.3f}")
print(f"강수량-대여량 r = {stats.pearsonr(day['rain'], day['count']).statistic:.3f}")
bands = pd.cut(day["temp"], [-20, 0, 10, 20, 25, 28, 34])
print(day.groupby(bands, observed=True)["count"].agg(["size", "mean"]).round(0))

plt.scatter(day["temp"], day["count"], s=10, color="#9a9a9a")
hot = day["temp"] > 28
plt.scatter(day.loc[hot, "temp"], day.loc[hot, "count"], s=14, color="#C8234B")
plt.xlabel("하루 평균기온 (°C)")
plt.ylabel("하루 대여량 (대)")
plt.show()

# ### EQ-1
s, n = 9951, 352
print("표준오차:", round(s / np.sqrt(n)))
print("표본을 4배로:", round(s / np.sqrt(4 * n)))

# ### EQ-2
for n in [1000, 4000, 2_400_000]:
    print(n, "명: ±", round(1.96 * np.sqrt(0.5 * 0.5 / n) * 100, 2), "%p")

# ### EQ-3
m1, s1, n1 = 18539, 10025, 270      # 맑은 날
m2, s2, n2 = 14034, 8920, 82        # 비 온 날
se = np.sqrt(s1**2 / n1 + s2**2 / n2)
print("분모(표준오차):", round(se), " t =", round((m1 - m2) / se, 2))
print("이름표를 섞은 세계의 표준오차:", round(9951 * np.sqrt(1 / n1 + 1 / n2)))

# ### EQ-4
sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
print("합동 표준편차:", round(sp), " d =", round((m1 - m2) / sp, 2))

# ### MP-0
# ☆ 포털에서 받은 원자료를 강의 자료와 같은 모양으로 합치기 (강의 자료의 파일을 쓰면 건너뜁니다)
import glob
SUB_RAW = sorted(glob.glob("서울교통공사_역별*승하차인원*.csv"))   # 공공데이터포털에서 받은 파일
ASOS_RAW = sorted(glob.glob("OBS_ASOS_TIM_*.csv"))                 # 기상자료개방포털에서 받은 파일
if not SUB_RAW and not ASOS_RAW:
    print("포털에서 받은 파일이 없습니다. 강의 자료의 파일 두 개로 1단계를 하십시오.")
if SUB_RAW:
    parts = []
    for f in SUB_RAW:
        t = pd.read_csv(f, encoding="cp949")                        # 포털 파일의 한글은 cp949
        t = t.rename(columns={"수송일자": "날짜", "승하차구분": "구분"})
        t.columns = [c.replace("시간대", "시") for c in t.columns]   # 13-14시간대 → 13-14시
        late = [c for c in ["23-24시", "24시이후"] if c in t.columns]   # 늦은 시간 열은 해마다 다르다
        if late:
            t["23시이후"] = t[late].sum(axis=1)
            t = t.drop(columns=late)
        parts.append(t)
    raw = pd.concat(parts).drop_duplicates(["날짜", "역번호", "구분"], keep="last")  # 기간이 겹치는 파일
    hours = [c for c in raw.columns if c.endswith(("시", "이전", "이후"))]
    city = raw.groupby(["날짜", "구분"])[hours].sum()
    city["합계"] = city.sum(axis=1)
    city.reset_index().to_csv("지하철_직접_합친_파일.csv", index=False)
    print("지하철", len(SUB_RAW), "개 파일 →", city.shape, city.index[0][0], "~", city.index[-1][0])
if ASOS_RAW:
    wr = pd.concat([pd.read_csv(f, encoding="cp949") for f in ASOS_RAW])
    wr = wr.drop_duplicates("일시", keep="last").sort_values("일시")   # 기간이 겹치면 한 번만
    wr[["지점", "지점명", "일시", "기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)"]].to_csv(
        "날씨_직접_합친_파일.csv", index=False)
    print("날씨", len(ASOS_RAW), "개 파일 →", len(wr), "시각", wr["일시"].iloc[0], "~", wr["일시"].iloc[-1])

# ### MP-1
# 데이터 바꾸기: MP-0으로 직접 합친 파일을 쓸 때는 이 두 줄만 고칩니다.
SUB = "서울교통공사_서울전체_일별_시간대별_승하차_2023_2024.csv"
WEATHER = "asos_서울108_시간자료_2023_2026.csv"

sub = pd.read_csv(SUB, parse_dates=["날짜"])
print(sub.shape, sub["날짜"].min().date(), "~", sub["날짜"].max().date(), "/ 빈칸", sub.isna().sum().sum())
on = sub[sub["구분"] == "승차"].set_index("날짜")             # 하루에 지하철을 탄 사람(승차)
w = pd.read_csv(WEATHER, parse_dates=["일시"])
w["강수량(mm)"] = w["강수량(mm)"].fillna(0)                    # 비가 안 온 시각은 빈칸으로 적혀 있다
rain = w.groupby(w["일시"].dt.normalize())["강수량(mm)"].sum()  # 하루 강수량
month = (on[["13-14시", "합계"]] / 10000).groupby(on.index.to_period("M")).mean().round(1)
print(month.loc["2024-02":"2024-06"])                         # 만 명, 하루 평균

month.index = month.index.strftime("%y.%m")
month["13-14시"].plot(marker="o", color="#C8234B", rot=0, figsize=(7, 3))
plt.ylabel("13~14시 승차 (만 명)")
plt.xlabel("")
plt.show()

# ### MP-2
import holidays
kr = holidays.KR(years=[2023, 2024])                          # 설·추석 연휴, 대체 공휴일까지
d = (on["합계"] / 10000).to_frame("승차").join(rain.rename("강수량"), how="inner")
d = d[d.index.to_period("M") != pd.Period("2024-04")]         # 13~14시 칸이 이상한 달은 뺀다
d["평일"] = [t.weekday() < 5 and t not in kr for t in d.index]
wk = d[d["평일"]].copy()                                       # 공휴일이 아닌 월~금
wk["비"] = wk["강수량"] >= 10                                   # 미리 정한 기준: 하루 10mm 이상
a, b = wk.loc[wk["비"], "승차"], wk.loc[~wk["비"], "승차"]
print("평일", len(wk), "일 가운데 비 온 날", len(a), "일")
print(f"비 온 평일 {a.mean():.1f}만 명, 그 밖의 평일 {b.mean():.1f}만 명, 차이 {a.mean() - b.mean():.1f}만 명")
print("p =", f"{stats.ttest_ind(a, b, equal_var=False).pvalue:.6f}")
g = wk.groupby(wk.index.month).agg(비온날=("비", "sum"), 승차=("승차", "mean"))
print(g.round(1).T.to_string())
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(g.index, g["승차"], color=["#C8234B" if m in (7, 8) else "#9a9a9a" for m in g.index])
ax.set_ylim(440, 520)
ax.set_xticks(g.index)
ax.set_xlabel("월")
ax.set_ylabel("평일 하루 승차 (만 명)")
plt.show()

# ### MP-3
wk["보통"] = wk.groupby([wk.index.month, wk.index.dayofweek])["승차"].transform("median")
wk["차이(%)"] = (wk["승차"] / wk["보통"] - 1) * 100            # 같은 달·같은 요일의 보통 날과 견준 차이
x, y = wk.loc[wk["비"], "차이(%)"], wk.loc[~wk["비"], "차이(%)"]
res = stats.ttest_ind(x, y, equal_var=False)
ci = res.confidence_interval(0.95)
print(f"비 온 평일 {x.mean():.2f}%, 그 밖의 평일 {y.mean():.2f}%")
print(f"웰치 t검정 p = {res.pvalue:.4f}, 차이의 95% 신뢰구간 {ci.low:.2f} ~ {ci.high:.2f}%p")
rng = np.random.default_rng(42)
v, lab = wk["차이(%)"].to_numpy(), wk["비"].to_numpy()
diffs = np.empty(10_000)
for i in range(10_000):
    s = rng.permutation(lab)                                  # 비 이름표를 섞는다
    diffs[i] = v[s].mean() - v[~s].mean()
print("순열검정 p =", (np.abs(diffs) >= abs(x.mean() - y.mean())).mean())
sp = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2))
print(f"효과 크기 d = {(y.mean() - x.mean()) / sp:.2f}")

plt.hist(y, bins=30, color="#9a9a9a", alpha=0.8, density=True, label="그 밖의 평일")
plt.hist(x, bins=15, color="#C8234B", alpha=0.6, density=True, label="비 온 평일")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("같은 달·같은 요일의 보통 날과 견준 차이 (%)")
plt.ylabel("밀도")
plt.legend()
plt.show()

# ### MP-4
usual = wk["보통"].median()
gap = (x.mean() - y.mean()) / 100 * usual
print(f"보통 평일 승차 {usual:.1f}만 명, 비 오는 평일에 줄어드는 사람 약 {abs(gap):.1f}만 명")
wk["비1"] = wk["강수량"] >= 1                                  # 다른 기준(1mm)으로 한 번 더: 검사 횟수 2번
x1, y1 = wk.loc[wk["비1"], "차이(%)"], wk.loc[~wk["비1"], "차이(%)"]
print(f"1mm 기준: 비 온 평일 {len(x1)}일 {x1.mean():.2f}%, 그 밖 {y1.mean():.2f}%, "
      f"p = {stats.ttest_ind(x1, y1, equal_var=False).pvalue:.3f}")

# ### EX-6
winter = day[day["season"] == "Winter"].copy()
winter["snowy"] = winter["date"].isin(h.loc[h["snow"] > 0, "date"])
a = winter.loc[winter["snowy"], "count"]
b = winter.loc[~winter["snowy"], "count"]
res = stats.ttest_ind(b, a, equal_var=False)
print(f"눈 온 날 {len(a)}일 {a.mean():,.0f}대, 안 온 날 {len(b)}일 {b.mean():,.0f}대")
print(f"웰치 t = {res.statistic:.2f}, p = {res.pvalue:.4f}")
rng = np.random.default_rng(42)
v, n_a = winter["count"].to_numpy(), len(a)
diffs = []
for _ in range(10_000):
    s = rng.permutation(v)
    diffs.append(s[n_a:].mean() - s[:n_a].mean())
print("순열검정 p =", (np.abs(diffs) >= abs(b.mean() - a.mean())).mean())
