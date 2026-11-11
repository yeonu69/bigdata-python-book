# 12장. 작년 이맘때처럼: 시계열 분석 (v3) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 12-k' 조각 하나가 책의 [파이썬 실습 12-k] 하나입니다. 본문 실습 12-5~12-9는 한국 전력 수요(2021~2025),
# MP-k는 [미니 프로젝트 12](독일 전력)의 단계, EQ-k는 04절 예제 확인, EX-더1은 [더 풀어 보기] 정답 코드입니다.
# 실행: python3 practice_ch12.py  (필요 라이브러리: pandas, numpy, matplotlib, scikit-learn, statsmodels, prophet, holidays)
# DATA 조각은 한국 전력 파일이 없으면 이 저장소에서 받아 옵니다(12-5보다 먼저 실행).

# ### 12-1
import numpy as np, pandas as pd                     # 숫자 계산과 표
import matplotlib.pyplot as plt                      # 그래프
from statsmodels.tsa.seasonal import seasonal_decompose   # 시계열 분해
from statsmodels.tsa.stattools import adfuller            # 정상성 검정
from statsmodels.tsa.statespace.sarimax import SARIMAX    # (계절) ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # 홀트-윈터스
from prophet import Prophet                          # 달력을 읽는 예측 도구
from sklearn.ensemble import RandomForestRegressor   # 랜덤 포레스트
import holidays                                      # 나라별 공휴일 달력

URL_AIR = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/airline-passengers.csv")
air = pd.read_csv(URL_AIR, parse_dates=["Month"], index_col="Month")
y = air["Passengers"].asfreq("MS").astype(float)   # 매달 첫날(MS) 간격
print(len(y), "개월:", y.index[0].strftime("%Y-%m"), "~", y.index[-1].strftime("%Y-%m"))

def mae(a, b):                                      # 평균 절대 오차 (식 12-5)
    return round(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))), 1)
year_mean = y.groupby(y.index.year).mean()          # 해마다 평균
print("1949년 평균", round(year_mean[1949], 1), "→ 1960년 평균", round(year_mean[1960], 1))
print("해마다 가장 붐빈 달:", y.groupby(y.index.year).idxmax().dt.month.value_counts().to_dict())

k = np.r_[0.5, np.ones(11), 0.5] / 12             # 무게 13개, 양 끝은 절반 (합은 1)
trend = y.rolling(13, center=True).apply(lambda w: w @ k)  # 창을 밀며 무게와 내적
print("추세 1949-07:", round(trend["1949-07-01"], 1), " 1960-06:", round(trend["1960-06-01"], 1))

plt.figure(figsize=(7, 3))
plt.plot(y, color="#8EC5E8", label="승객 수")
plt.plot(trend, color="#E04B3C", label="12개월 이동평균(추세)")
plt.ylabel("승객(천 명)"); plt.legend(); plt.show()

# ### 12-2
dec = seasonal_decompose(y, model="multiplicative", period=12)  # 추세 × 계절 × 나머지
season = dec.seasonal.iloc[:12]                     # 계절 지수는 해마다 같으니 열두 달만
season.index = season.index.month                   # 날짜 대신 1~12월로
print("계절 지수 1월", round(season[1], 3), "/ 7월", round(season[7], 3), "/ 11월", round(season[11], 3))

ly = np.log(y)                                      # 로그: 흔들림의 크기를 고르게
for name, s in {"원자료": y, "로그": ly, "로그+1차 차분": ly.diff(),
                "로그+1차+계절 차분": ly.diff().diff(12)}.items():
    print(name, "ADF p값", round(adfuller(s.dropna(), autolag="AIC")[1], 4))

season.plot.bar(color=["#E04B3C" if m in (7, 8) else "#8EC5E8" for m in season.index],
                figsize=(7, 2.6), rot=0)
plt.axhline(1, color="gray", lw=1)                  # 1보다 크면 그해 흐름보다 붐비는 달
plt.xlabel("달"); plt.ylabel("계절 지수"); plt.show()

# ### 12-3
train, test = y.loc[:"1958"], y.loc["1959":]       # 1958년까지 학습, 1959~1960년 시험
h = len(test)                                       # 내다볼 달 수
print("학습", len(train), "개월 / 시험", h, "개월")

slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)   # 첫 값과 끝 값을 이은 기울기
base = {"평균": np.repeat(train.mean(), h),                   # 지금까지의 평균
        "직전 값": np.repeat(train.iloc[-1], h),               # 마지막 값 그대로
        "추세(drift)": train.iloc[-1] + slope * np.arange(1, h + 1),  # 기울기대로 연장
        "작년 같은 달": np.tile(train.iloc[-12:].values, 2)}     # 1958년 열두 달을 두 번

score = pd.Series({k: mae(test, f) for k, f in base.items()})
print(score.to_string())
score.plot.barh(color=["#8EC5E8"] * 3 + ["#E04B3C"], figsize=(6, 2.6))
plt.xlabel("평균 오차 MAE(천 명)"); plt.show()

# ### 12-4
def airline(series):                                # 항공사 모델: 로그 + 계절 ARIMA
    return SARIMAX(np.log(series), order=(0, 1, 1),
                   seasonal_order=(0, 1, 1, 12)).fit(disp=False)
def holt(series):                                   # 홀트-윈터스: 추세 + 곱셈 계절
    return ExponentialSmoothing(series, trend="add", seasonal="mul",
                                seasonal_periods=12).fit()

sar = airline(train)
p_sar = np.exp(sar.forecast(h))                     # 로그로 배웠으니 exp로 되돌린다
p_hw = holt(train).forecast(h)
print("항공사 모델 MAE", mae(test, p_sar), "/ 홀트-윈터스 MAE", mae(test, p_hw))
ci = np.exp(sar.get_forecast(h).conf_int(alpha=0.05))   # 95% 예측 구간(아래, 위)
inside = ((test >= ci.iloc[:, 0]) & (test <= ci.iloc[:, 1])).sum()
print("95% 구간 안에 든 달:", inside, "/", h)

rows = []
for Y in range(1955, 1961):                         # 1955~1960년을 차례로 시험
    tr, te = y.loc[:str(Y - 1)], y.loc[str(Y)]      # 그 전해까지 학습, 그해 시험
    f_sn = tr.iloc[-12:].values                     # 작년 같은 달
    f_sar = np.exp(airline(tr).forecast(12)).values
    f_hw = holt(tr).forecast(12).values
    rows.append({"시험 해": Y, "작년 같은 달": mae(te, f_sn), "SARIMA": mae(te, f_sar),
                 "홀트-윈터스": mae(te, f_hw), "둘의 평균": mae(te, (f_sar + f_hw) / 2)})
roll = pd.DataFrame(rows).set_index("시험 해")
print("해마다 1등:", roll.iloc[:, :3].idxmin(axis=1).value_counts().to_dict())
print("여섯 해 평균:", roll.mean().round(1).to_dict())

plt.figure(figsize=(7, 3))
plt.fill_between(test.index, ci.iloc[:, 0], ci.iloc[:, 1],      # 아래·위 끝 사이를 칠한다
                 color="#8EC5E8", alpha=0.4, label="95% 구간")
plt.plot(test, color="black", label="실제")
plt.plot(p_sar, color="#E04B3C", label="항공사 모델")
plt.plot(p_hw, color="#4466AE", ls="--", label="홀트-윈터스")
plt.gcf().autofmt_xdate()                           # 날짜 글자를 비스듬히 눕혀 겹치지 않게
plt.ylabel("승객(천 명)"); plt.legend(ncol=2); plt.show()

# ### DATA
import os, urllib.parse, urllib.request
KPX = "kpx_시간별_전국_전력수요량_2021_2025.csv"
if not os.path.exists(KPX):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/yeonu69/bigdata-python-book/main/ch12/data/" + urllib.parse.quote(KPX), KPX)
print(os.path.getsize(KPX), "bytes")

# ### 12-5
DATA = ""                                           # 파일을 둔 폴더(코랩 파일 창에 올렸으면 빈 글자)
kp = pd.read_csv(DATA + "kpx_시간별_전국_전력수요량_2021_2025.csv")
print(kp.shape, "/ 빈칸이 있는 줄:", kp.isna().any(axis=1).sum())
kp = kp.dropna(subset=["날짜"])                      # 날짜까지 빈 줄은 버린다
kp["날짜"] = pd.to_datetime(kp["날짜"])
kp = kp.set_index("날짜").asfreq("D")                # 날짜를 인덱스로, 빠진 날은 빈 줄로 드러낸다
print("같은 날짜:", kp.index.duplicated().sum(), "/ 날짜 순서대로:", kp.index.is_monotonic_increasing,
      "/ 남은 빈칸:", kp.isna().sum().sum())

c = kp.mean(axis=1) * 24 / 1000                     # 하루 GWh = 24시간 평균(MW) × 24 ÷ 1000
print(len(c), "일 / 하루 평균", round(c.mean(), 1), "GWh")
print("해마다 평균:", c.groupby(c.index.year).mean().round(1).to_dict())
print("가장 많은 날\n" + c.nlargest(3).round(1).to_string(header=False))
print("가장 적은 날\n" + c.nsmallest(3).round(1).to_string(header=False))
d = c.idxmax()
print("그날 가장 많이 쓴 시각:", kp.loc[d].idxmax(), int(kp.loc[d].max()), "MW")

c.plot(color="#8EC5E8", lw=0.6, figsize=(7, 3), label="하루 수요")
c.rolling(28, center=True).mean().plot(color="#4466AE", label="4주 이동평균")   # 요일 톱니를 지운다
plt.scatter([c.idxmax(), c.idxmin()], [c.max(), c.min()], color="#E04B3C", zorder=3)
plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1))
plt.ylabel("GWh"); plt.xlabel(""); plt.show()

# ### 12-6
dow = c.groupby(c.index.dayofweek).mean()          # 0=월요일 … 6=일요일
print("요일 평균(월~일):", dow.round(1).tolist())
print("주말은 평일보다", round((1 - dow[[5, 6]].mean() / dow[:5].mean()) * 100, 1), "% 적다")
mon = c.groupby(c.index.month).mean()
print("가장 많은 달", mon.idxmax(), "월", round(mon.max(), 1), "/ 가장 적은 달", mon.idxmin(), "월", round(mon.min(), 1))

lags = [1, 2, 3, 4, 5, 6, 7, 14, 364, 365]
r = pd.Series({k: c.autocorr(k) for k in lags})   # k칸 민 자기 자신과의 상관계수
print("자기상관:", r.round(3).to_dict())

r.plot.bar(color=["#E04B3C" if k in (7, 364) else "#8EC5E8" for k in r.index],
           figsize=(7, 3), rot=0)
plt.xlabel("시차(일)"); plt.ylabel("자기상관"); plt.show()

# ### 12-7
tr_e, te_e = c.loc[:"2024"], c.loc["2025"]          # 2024년까지 배우고 2025년 시험
f_364 = np.tile(tr_e.iloc[-364:].values, 2)[:365]   # 2024년의 마지막 52주를 두 번 잇는다
leak = c.shift(364).loc["2025"]                     # 흔한 실수: 시험 해의 값이 섞인다
print("평균으로 찍기 MAE", mae(te_e, tr_e.loc["2024"].mean()))
print("작년 같은 요일 MAE", mae(te_e, f_364), " (shift(364)로 하면", mae(te_e, leak), ")")

def fit_prophet(series, holidays_on, days=365):
    df = series.reset_index()
    df.columns = ["ds", "y"]                        # 프로펫은 열 이름이 ds, y여야 한다
    m = Prophet(daily_seasonality=False)
    if holidays_on:
        m.add_country_holidays(country_name="KR")   # 한국 공휴일 달력을 알려 준다
    m.fit(df)
    return m, m.predict(m.make_future_dataframe(periods=days)).set_index("ds")

m0, fc0 = fit_prophet(tr_e, False)
m1, fc1 = fit_prophet(tr_e, True)
p0, p1 = fc0["yhat"].loc["2025"], fc1["yhat"].loc["2025"]
print("프로펫 기본 MAE", mae(te_e, p0), "/ 프로펫 + 휴일 MAE", mae(te_e, p1))
err = (te_e - p1).abs().sort_values(ascending=False)   # 틀린 크기를 큰 것부터
print("크게 틀린 날:", [str(d.date()) for d in err.index[:3]])
d0 = err.index[0]
print(d0.date(), "실제", round(te_e[d0], 1), "/ 예측의 부분들:",
      fc1.loc[d0, ["trend", "weekly", "yearly", "holidays", "yhat"]].round(1).to_dict())

oct25 = slice("2025-10-01", "2025-10-31")
te_e[oct25].plot(color="black", label="실제", figsize=(7, 3))
p1[oct25].plot(color="#E04B3C", label="프로펫 + 휴일")
sn25 = pd.Series(f_364, te_e.index)                # 배열에 2025년 날짜를 붙인다
sn25[oct25].plot(color="#8EC5E8", label="작년 같은 요일")
plt.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))  # 범례는 그래프 위에
plt.ylabel("GWh"); plt.xlabel(""); plt.show()

# ### 12-8
rows = []
for Y in range(2023, 2026):                         # 2023~2025년을 차례로 시험
    tr, te = c.loc[:str(Y - 1)], c.loc[str(Y)]      # 그 전해까지 학습, 그해 시험
    sn = np.tile(tr.iloc[-364:].values, 2)[:len(te)]  # 작년 같은 요일 (시험 해 값은 쓰지 않음)
    p = fit_prophet(tr, True, len(te))[1]["yhat"].loc[str(Y)].values
    rows.append({"시험 해": Y, "배운 해 수": Y - 2021, "작년 같은 요일": mae(te, sn),
                 "프로펫+휴일": mae(te, p), "둘의 평균": mae(te, (sn + p) / 2)})

eroll = pd.DataFrame(rows).set_index("시험 해")      # 사전의 목록 → 표
print(eroll.to_string())
print(eroll.iloc[:, 1:].agg(["mean", "max"]).round(1).to_string())

eroll.iloc[:, 1:].plot.bar(color=["#8EC5E8", "#E04B3C", "#4466AE"], figsize=(7, 3), rot=0)
plt.ylabel("MAE(GWh)"); plt.xlabel("시험 해"); plt.show()

# ### 12-9
kr = holidays.KR(years=range(2021, 2026))          # 한국 공휴일(설·추석 연휴, 대체 휴일 포함)
hol = pd.Series([d in kr for d in c.index.date], index=c.index).astype(int)

X = pd.DataFrame({"어제": c.shift(1), "지난주 같은 요일": c.shift(7),
                  "작년 같은 요일": c.shift(364), "지난 7일 평균": c.shift(1).rolling(7).mean(),
                  "달": c.index.month, "요일": c.index.dayofweek,
                  "휴일": hol, "어제가 휴일": hol.shift(1, fill_value=0)})
data = X.join(c.rename("y")).dropna()               # 정답 열을 붙이고 빈칸 줄을 버린다

trB, teB = data.loc[:"2024"], data.loc["2025"]
feats = [k for k in data.columns if k != "y"]       # 정답(y)을 뺀 나머지가 입력
print("어제 값 MAE", mae(teB.y, teB["어제"]), "/ 지난주 같은 요일 MAE", mae(teB.y, teB["지난주 같은 요일"]))
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf.fit(trB[feats], trB.y)                           # 2024년까지로 배운다
pred = rf.predict(teB[feats])                       # 2025년 매일의 다음 날
print("랜덤 포레스트 MAE", mae(teB.y, pred), "/ 하루 수요의", round(mae(teB.y, pred) / teB.y.mean() * 100, 1), "% / 배운 날", len(trB))

imp = pd.Series(rf.feature_importances_, feats).sort_values()   # 기여한 몫
imp.plot.barh(color="#E04B3C", figsize=(6, 3)); plt.xlabel("중요도"); plt.show()

# ### EX-더1
no_hol = [k for k in feats if "휴일" not in k]       # 휴일 두 열을 뺀다
rf2 = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf2.fit(trB[no_hol], trB.y)
pred2 = rf2.predict(teB[no_hol])
is_hol = (teB["휴일"] == 1).values
print("한 해 전체 MAE — 휴일 있음", mae(teB.y, pred), "/ 없음", mae(teB.y, pred2))
print("휴일", is_hol.sum(), "일의 MAE — 휴일 있음", mae(teB.y[is_hol], pred[is_hol]),
      "/ 없음", mae(teB.y[is_hol], pred2[is_hol]))

# ### EQ-1
w = np.array([1, 1, 1]) / 3                         # 무게(커널) 세 개, 합은 1
print(round(np.array([112, 118, 132]) @ w, 1))      # 내적 = 3개월 이동평균

# ### EQ-2
s = pd.Series([10, 20, 10, 20, 10])
print("시차 1:", round(s.autocorr(1), 3), " 시차 2:", round(s.autocorr(2), 3))

# ### EQ-3
s_t = 112                                           # 첫 평활값
for v in [118, 132]:
    s_t = 0.3 * v + 0.7 * s_t                       # (식 12-4)
    print(round(s_t, 2))

# ### EQ-4
real = np.array([360, 342, 406])
for name, f in [("작년 같은 달", [340, 318, 362]), ("항공사 모델", [348.6, 331.6, 383.2])]:
    e = np.abs(real - np.array(f))
    print(f"{name}  MAE {e.mean():.1f}  MAPE {(e / real).mean() * 100:.1f}%")

# ### MP-1
URL_EL = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
el = pd.read_csv(URL_EL, parse_dates=["Date"], index_col="Date").asfreq("D")
g = el["Consumption"]                                # 하루 전력 소비, 단위 GWh
print(el.shape, "빈칸:", el.isna().sum().to_dict())
print("첫 값이 있는 날:", {k: str(el[k].first_valid_index().date()) for k in el.columns})

# ### MP-2
gy = g.groupby(g.index.year).mean()                  # 해마다 하루 평균
print("전년 대비 변화(%):", (gy.pct_change() * 100).round(1).loc[2009:2015].to_dict())
gm = g.resample("MS").mean()                         # 달마다 평균
ratio = (gm.loc["2014"].values / gm.loc["2013"].values - 1) * 100
print("2014년 달마다 전년 같은 달 대비(%):", np.round(ratio, 1))
gy.plot.bar(color=["#E04B3C" if t == 2014 else "#8EC5E8" for t in gy.index], figsize=(7, 3), rot=0)
plt.ylim(1200, 1420); plt.ylabel("하루 평균 수요(GWh)"); plt.xlabel(""); plt.show()

# ### MP-3
def fit_de(series, extra=None, country=True, days=365):
    df = series.reset_index(); df.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=False, holidays=extra)
    if country:
        m.add_country_holidays(country_name="DE")        # 독일 공휴일 달력
    m.fit(df)
    return m, m.predict(m.make_future_dataframe(periods=days)).set_index("ds")["yhat"].loc["2017"]
tr_g, te_g = g.loc[:"2016"], g.loc["2017"]
m1, q1 = fit_de(tr_g)
d0 = (te_g - q1).abs().idxmax()                      # 가장 크게 틀린 날
print("프로펫 + 휴일 MAE", mae(te_g, q1), "/ 가장 크게 틀린 날", d0.date(), "실제", round(te_g[d0], 1), "예측", round(q1[d0], 1))
print("배운 휴일 수:", len(m1.train_holiday_names), "/ 종교개혁 기념일 있나:", "Reformation Day" in list(m1.train_holiday_names))
days = [d for d in holidays.Germany(years=range(2006, 2018)) if d.weekday() < 5]   # 평일에 든 공휴일만
extra = pd.DataFrame({"holiday": "평일 공휴일", "ds": pd.to_datetime(sorted(days))})  # 모두 한 이름으로
m2, q2 = fit_de(tr_g, extra, country=False)          # 나라 달력 대신 이 표를 알려 준다
print("평일 공휴일 무리 MAE", mae(te_g, q2), "/ 10월 31일 예측", round(q2["2017-10-31"], 1))

# ### MP-4
for start in ["2015", "2013", "2011", "2006"]:       # 학습 시작 해를 바꾼다(끝은 늘 2016년)
    print(start, "년부터 배움: MAE", mae(te_g, fit_de(g.loc[start:"2016"])[1]))

# ### MP-5
print("자기상관 독일:", {k: float(round(g.autocorr(k), 3)) for k in [1, 7]}, "/ 한국:", {k: float(round(c.autocorr(k), 3)) for k in [1, 7]})
print("2017년 다음 날 MAE: 어제", mae(te_g, g.shift(1).loc["2017"]), "/ 지난주 같은 요일", mae(te_g, g.shift(7).loc["2017"]))
