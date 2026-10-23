# 03장. 지저분한 데이터 정리하기 (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 3-k' 조각 하나가 책의 [파이썬 실습 3-k] 하나입니다. MP-k는 [미니 프로젝트 3]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: pandas, numpy, matplotlib)
# 미니 프로젝트는 강의 자료의 '지하철_원자료_표본.zip'이 이 파일과 같은 폴더에 있어야 합니다.

# ### 3-1
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
plt.rcParams["axes.unicode_minus"] = False

URL = ("https://raw.githubusercontent.com/syedsharin/"
       "Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv")
try:
    pd.read_csv(URL)                        # 인코딩을 말하지 않으면 UTF-8로 읽는다
except UnicodeDecodeError as e:
    print("읽기 실패:", e)
raw = pd.read_csv(URL, encoding="latin-1")  # 원본: 이 표는 끝까지 고치지 않는다
print(raw.shape)

names = {"Date": "date", "Rented Bike Count": "count", "Hour": "hour",
         "Temperature(°C)": "temp", "Humidity(%)": "humidity",
         "Wind speed (m/s)": "wind", "Visibility (10m)": "visibility",
         "Dew point temperature(°C)": "dew", "Solar Radiation (MJ/m2)": "solar",
         "Rainfall(mm)": "rain", "Snowfall (cm)": "snow", "Seasons": "season",
         "Holiday": "holiday", "Functioning Day": "functioning"}
df = raw.rename(columns=names)               # 고친 것은 새 이름 df에 담는다
print(df[["date", "hour", "count", "temp"]].head(3))
print("date 열의 자료형:", df["date"].dtype)
print(df[["count", "temp", "humidity", "dew", "rain"]].describe().round(1).T[["mean", "min", "50%", "max"]])

df["humidity"].plot.hist(bins=50, color="#8EC5E8", edgecolor="white", figsize=(6, 3))
plt.xlabel("습도(%)")
plt.ylabel("시간 수")
plt.show()

# ### 3-2
wrong = pd.to_datetime(df["date"], format="mixed")      # 칸마다 따로 짐작
good = pd.to_datetime(df["date"], format="%d/%m/%Y")    # 형식을 분명히 알려 줌
print("짐작한 기간:", wrong.min().date(), "~", wrong.max().date())
print("알려 준 기간:", good.min().date(), "~", good.max().date())
print("다른 날로 옮겨 간 날:", (wrong != good).sum() // 24, "일")
print(pd.crosstab(wrong.dt.month, df["season"]).loc[[1, 7]] // 24)

cmp = pd.DataFrame({"바르게": df.groupby(good.dt.month)["count"].mean() * 24,
                    "잘못": df.groupby(wrong.dt.month)["count"].mean() * 24}).round()
print(cmp.T.to_string())
print("가장 많은 달 ÷ 가장 적은 달:", (cmp.max() / cmp.min()).round(1).to_dict())
df["date"] = good                           # 이제부터는 바르게 읽은 날짜를 쓴다

cmp.plot.bar(color=["#3F7CC0", "#F2A65A"], figsize=(7, 3))
plt.xlabel("달")
plt.ylabel("하루 평균 대여량(대)")
plt.xticks(rotation=0)
plt.show()

# ### 3-3
off = df[df["functioning"] == "No"]         # 운영하지 않은 시간
print("운영 안 한 시간:", len(off), " 날짜 수:", off["date"].nunique())
print("대여량이 0인 시간:", (df["count"] == 0).sum())
print("달마다:", off.groupby(off["date"].dt.month).size().to_dict())

zero = df["humidity"] == 0
print("습도 0%:", zero.sum(), "시간", df.loc[zero, "date"].dt.strftime("%m-%d").unique().tolist())
# 기온과 이슬점으로 습도를 다시 계산해 본다 (마그누스 근사식)
calc = 100 * np.exp(17.625 * df["dew"] / (243.04 + df["dew"])) \
           / np.exp(17.625 * df["temp"] / (243.04 + df["temp"]))
print("습도가 0이 아닌 시간: 기록과 계산의 차이 평균",
      round((calc - df["humidity"])[~zero].abs().mean(), 2), "%p")
print("습도 0%인 시간의 계산값:", round(calc[zero].min()), "~", round(calc[zero].max()), "%")
w = calc[zero].idxmax()                          # 계산값이 가장 큰 시간
print("예:", df.loc[w, "date"].date(), df.loc[w, "hour"], "시, 기온", df.loc[w, "temp"], "이슬점", df.loc[w, "dew"], "→", round(calc[w]), "%")
df["humidity"] = df["humidity"].replace(0, np.nan)   # 0% → 결측(NaN)
print("습도 빈칸:", df["humidity"].isna().sum())

plt.figure(figsize=(5, 4))
plt.scatter(calc[~zero], raw.loc[~zero, "Humidity(%)"], s=3, color="#8EC5E8", label="보통 시간")
plt.scatter(calc[zero], raw.loc[zero, "Humidity(%)"], s=30, color="#E04B3C", label="습도 0%")
plt.xlabel("기온·이슬점으로 계산한 습도(%)")
plt.ylabel("기록된 습도(%)")
plt.legend()
plt.show()

# ### 3-4
h = df[df["functioning"] == "Yes"].copy()   # 운영한 시간만
q1, q3 = h["count"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
top = h.loc[h["count"].idxmax()]
print("가장 많은 시간:", top["date"].date(), top["hour"], "시", top["count"], "대")
print(f"운영한 {len(h):,}시간  Q1 {q1:.0f}, Q3 {q3:,.0f}, IQR {iqr:.0f}, 위 울타리 {upper:,.0f}")
out = h[h["count"] > upper]
print("이상치 후보:", len(out), "시간, 많은 시각:", out["hour"].value_counts().head(2).to_dict())
print("그 가운데 평일:", round((out["date"].dt.dayofweek < 5).mean() * 100, 1), "%")

q = h.groupby("hour")["count"].quantile([0.25, 0.75]).unstack()   # 시각마다 Q1, Q3
fence = q[0.75] + 1.5 * (q[0.75] - q[0.25])                      # 시각마다 위 울타리
print(f"저녁 6시끼리: Q3 {q.loc[18, 0.75]:,.0f}, 위 울타리 {fence[18]:,.0f}")
print("시각별로 잰 이상치 후보:", (h["count"] > h["hour"].map(fence)).sum(), "시간")
print("비가 0인 시간:", round((h["rain"] == 0).mean() * 100, 1), "%,  비의 IQR:",
      h["rain"].quantile(0.75) - h["rain"].quantile(0.25), ",  비 온 시간:", (h["rain"] > 0).sum())

h.boxplot(column="count", by="hour", figsize=(8, 3.5), grid=False)
plt.axhline(upper, color="#E04B3C", linestyle="--")
plt.title("")
plt.suptitle("")
plt.xlabel("시각")
plt.ylabel("한 시간 대여량(대)")
plt.show()

# ### 3-5
day = (h.groupby("date")
        .agg(count=("count", "sum"), temp=("temp", "mean"), rain=("rain", "sum"),
             snow=("snow", "sum"), humidity=("humidity", "mean"),
             season=("season", "first"), holiday=("holiday", "first"),
             hours=("hour", "size"))
        .reset_index())
print(len(day), "일 →", end=" ")
day = day[day["hours"] == 24].drop(columns="hours").reset_index(drop=True)
print(len(day), "일")

all365 = df.groupby("date")["count"].sum()               # 0을 그대로 둔 365일
print(f"하루 평균  0 포함 {all365.mean():,.0f}대,  운영한 날만 {day['count'].mean():,.0f}대")
m0 = all365.groupby(all365.index.month).mean()
m1 = day.groupby(day["date"].dt.month)["count"].mean()
print(f"9월  0 포함 {m0[9]:,.0f}대,  운영한 날만 {m1[9]:,.0f}대")
print("바쁜 달 순서  0 포함:", m0.sort_values(ascending=False).index[:4].tolist(),
      " 운영한 날만:", m1.sort_values(ascending=False).index[:4].tolist())
top, low = day.loc[day["count"].idxmax()], day.loc[day["count"].idxmin()]
print("가장 많은 날:", top["date"].date(), top["count"], " 가장 적은 날:", low["date"].date(), low["count"], "비", low["rain"], "mm")

pd.DataFrame({"0 포함": m0, "운영한 날만": m1}).plot.bar(color=["#B9C0C8", "#E04B3C"], figsize=(7, 3))
plt.xlabel("달")
plt.ylabel("하루 평균 대여량(대)")
plt.xticks(rotation=0)
plt.show()

# ### 3-6
hol = pd.DataFrame({
    "date": ["2017-12-25", "2018-01-01", "2018-02-15", "2018-02-16", "2018-02-17",
             "2018-03-01", "2018-05-05", "2018-05-07", "2018-05-22", "2018-06-06",
             "2018-06-13", "2018-08-15", "2018-09-23", "2018-09-24", "2018-09-25",
             "2018-09-26", "2018-10-03", "2018-10-09"],
    "name": ["성탄절", "신정", "설날 연휴", "설날", "설날 연휴", "삼일절", "어린이날",
             "대체공휴일", "부처님오신날", "현충일", "지방선거일", "광복절", "추석 연휴",
             "추석", "추석 연휴", "대체공휴일", "개천절", "한글날"]})
hol["date"] = pd.to_datetime(hol["date"])               # 연-월-일 글자라 헷갈릴 일이 없다
day = day.merge(hol, on="date", how="left")
day["official"] = day["name"].notna()
print(len(day), "행")
print(pd.crosstab(day["holiday"], day["official"]))
diff = day[(day["holiday"] == "Holiday") != day["official"]]
print(diff[["date", "holiday", "name", "count"]].to_string(index=False))
for label, is_hol in [("원래 열", day["holiday"] == "Holiday"), ("공식 표", day["official"])]:
    print(f"{label} 기준  공휴일 {day.loc[is_hol, 'count'].mean():,.0f}대,"
          f"  그 밖의 날 {day.loc[~is_hol, 'count'].mean():,.0f}대")

day["offday"] = (day["date"].dt.dayofweek >= 5) | day["official"]   # 주말이거나 공휴일
print(day.groupby("offday")["count"].agg(["size", "mean"]).round())
h["offday"] = (h["date"].dt.dayofweek >= 5) | h["date"].isin(hol["date"])   # 시간 표에도
print("아침 8시 평균:", h[h["hour"] == 8].groupby("offday")["count"].mean().round().to_dict())

day.to_csv("bike_daily_clean.csv", index=False, encoding="utf-8-sig")
h.to_csv("bike_hourly_clean.csv", index=False, encoding="utf-8-sig")
check = pd.read_csv("bike_daily_clean.csv", parse_dates=["date"])
print("다시 읽은 표:", check.shape, check["date"].min().date(), "~", check["date"].max().date())

h[h["hour"] == 8].groupby("offday")["count"].mean().rename({False: "평일", True: "쉬는 날"}) \
    .plot.bar(color=["#3F7CC0", "#E04B3C"], figsize=(4, 3))
plt.xlabel("")
plt.ylabel("아침 8시 평균 대여량(대)")
plt.xticks(rotation=0)
plt.show()

# ### 3-7
s = all365.where(~all365.index.isin(off["date"]))        # 운영하지 않은 13일 → NaN
print("빈칸:", s.isna().sum(), "일")
ways = {"지우기": s.dropna(), "0으로": s.fillna(0),
        "평균으로": s.fillna(s.mean()), "중앙값으로": s.fillna(s.median())}
print(pd.DataFrame({k: [len(v), v.mean(), v.median(), v.std()] for k, v in ways.items()},
                   index=["날수", "평균", "중앙값", "표준편차"]).round(0))
x = pd.Series([4, 6, 8, 10, 12])                       # [예제 3-3]의 다섯 날
print("판다스 std( ):", round(x.std(), 2), "  넘파이 np.std( ):", round(np.std(x), 2))

# ### 3-8
log = pd.DataFrame([
    ("인코딩", "latin-1로 읽음", "°C 바이트(0xB0)가 UTF-8이 아님", "14열"),
    ("날짜", "format='%d/%m/%Y'", "일/월/연 순서, 계절 열로 확인", "365일"),
    ("습도 0%", "결측(NaN)으로", "이슬점과 모순", "17시간"),
    ("운영 안 한 시간", "하루 분석에서 뺌", "0은 수요가 아니라 운영 중단", "295시간"),
    ("24시간 미만인 날", "뺌", "10월 6일은 17시간만 운영", "1일"),
    ("이상치 후보", "그대로 둠", "퇴근 시간의 진짜 값", "152시간"),
    ("공휴일", "공식 표를 합쳐 새 열로", "원래 열과 8일이 다름, 원래 열도 남김", "17일")],
    columns=["무엇을", "어떻게", "왜", "몇 건"])
log.to_csv("cleaning_log.csv", index=False, encoding="utf-8-sig")
print(log.to_string(index=False))

# ### 3-9
wide = df.pivot(index="date", columns="hour", values="count")   # 날짜 × 시각
print("넓은 표:", wide.shape)
print(wide.iloc[:2, :4])
long = wide.reset_index().melt(id_vars="date", var_name="hour", value_name="count")
print("깔끔한 표:", long.shape)
print(long.sort_values(["date", "hour"]).head(3).to_string(index=False))

# ### 3-10
df["humidity_calc"] = df["humidity"].isna()             # 채웠다는 표시
df["humidity"] = df["humidity"].fillna(calc.round())
print(df.loc[df["humidity_calc"], ["date", "hour", "temp", "dew", "humidity"]].head(3).to_string(index=False))
print("채운 시간:", df["humidity_calc"].sum())

# ### MP-1
import zipfile, glob
zipfile.ZipFile("지하철_원자료_표본.zip").extractall("subway_raw")   # 강의 자료 파일
raw_s = {}
for f in sorted(glob.glob("subway_raw/*.csv")):
    t = pd.read_csv(f, encoding="cp949")       # 공공데이터포털 파일의 한글은 cp949
    tag = f[-12:-4]                            # 파일 이름 끝의 기준일, 예: 20211231
    raw_s[tag] = t
    print(tag, t.shape, t.iloc[0, 1], "~", t.iloc[-1, 1], list(t.columns[1:3]), t.columns[-1],
          "빈칸", t.isna().sum().sum())

# ### MP-2
naive = {}
for tag, t in raw_s.items():                   # 정제하지 않고: 숫자 열을 모두 더한다
    num = t.select_dtypes("number").drop(columns=["연번", "역번호"], errors="ignore")
    naive[tag[:4]] = naive.get(tag[:4], 0) + num.sum().sum()
days = {"2021": 365, "2022": 365, "2023": 365, "2024": 366}
naive = pd.Series({y: naive[y] / days[y] for y in days}).round()
print("정제 전 하루 평균(세 역):", naive.astype(int).to_dict())

# ### MP-3
hours = ["06시이전"] + [f"{k:02d}-{k + 1:02d}시간대" for k in range(6, 23)] + ["23시이후"]
parts = []
for tag, t in raw_s.items():
    t = t.copy()
    if "합 계" in t.columns:                       # 2021년: 시간대 19칸 + 합계
        t = t.drop(columns="합 계")
    else:                                          # 2022년부터: 20칸 → 23시 이후를 한 칸으로
        t["23시이후"] = t["23-24시간대"] + t["24시이후"].fillna(0)   # 빈칸은 0으로 본다
        t = t.drop(columns=["23-24시간대", "24시이후"])
    t.columns = ["연번", "날짜", "호선", "역번호", "역명", "구분"] + hours
    t["호선"] = t["호선"].astype(str).str.replace("호선", "") + "호선"   # 2 → 2호선
    t["파일"] = tag
    parts.append(t)
sub = pd.concat(parts, ignore_index=True)
sub["날짜"] = pd.to_datetime(sub["날짜"])
print("이어 붙인 행:", len(sub))
key = ["날짜", "역명", "구분"]
print("같은 날짜·역·구분이 겹친 행:", sub.duplicated(key, keep=False).sum())
q3_23 = sub[(sub["날짜"] >= "2023-07-01") & (sub["날짜"] <= "2023-09-30")]
print("2023년 7~9월 합계:", q3_23.groupby("파일")[hours].sum().sum(axis=1).to_dict())
sub = sub.sort_values(["파일"] + key).drop_duplicates(key, keep="last")   # 가장 나중 파일을 믿는다
print("겹침을 지운 뒤:", len(sub), "행,", sub["날짜"].nunique(), "일")

# ### MP-4
sub["달"] = sub["날짜"].dt.strftime("%Y-%m")
ratio = (sub.groupby("달")["13-14시간대"].sum() / sub.groupby("달")["12-13시간대"].sum()).round(2)
print("13~14시 ÷ 12~13시가 가장 큰 달:", ratio.sort_values().tail(3).to_dict())
bad = sub["달"] == "2024-04"
sub["채움"] = bad                                   # 고친 칸이 있는 행 표시
sub.loc[bad, "13-14시간대"] = ((sub.loc[bad, "12-13시간대"] + sub.loc[bad, "14-15시간대"]) / 2).round()
sub["합계"] = sub[hours].sum(axis=1)
clean = sub.groupby(sub["날짜"].dt.year)["합계"].sum() / sub.groupby(sub["날짜"].dt.year)["날짜"].nunique()
clean.index = clean.index.astype(str)
result = pd.DataFrame({"정제 전": naive, "정제 후": clean.round()})
print(result.astype(int).to_string())
cols = ["날짜", "호선", "역번호", "역명", "구분"] + hours + ["합계", "채움", "파일"]
sub[cols].to_csv("지하철_3개역_정제_2021_2024.csv", index=False, encoding="utf-8-sig")

(result / 10000).plot.bar(color=["#B9C0C8", "#E04B3C"], figsize=(6, 3))
plt.ylabel("하루 평균 승하차(만 명)")
plt.xlabel("")
plt.xticks(rotation=0)
plt.show()

# ### EX-6
eight = h[h["hour"] == 8]["count"]                    # 아침 8시만
a, b = eight.quantile([0.25, 0.75])
print(f"아침 8시 {len(eight)}시간: Q1 {a:.0f}, Q3 {b:,.0f}, 위 울타리 {b + 1.5 * (b - a):,.1f}")
print("전체 기준으로 후보였던 8시:", (eight > upper).sum(), "시간,  8시 기준 후보:", (eight > b + 1.5 * (b - a)).sum(), "시간")
