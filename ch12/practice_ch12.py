# 12장. 작년 이맘때처럼: 시계열 분석 (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 12-k' 조각 하나가 책의 [파이썬 실습 12-k] 하나입니다. MP-k는 [미니 프로젝트 12]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: pandas, numpy, matplotlib, scikit-learn, statsmodels, prophet, holidays)

# ### 12-1
import warnings
warnings.filterwarnings("ignore")                  # 라이브러리의 경고 문구는 숨긴다
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib                    # 코랩: 한글 글꼴을 자동으로 설정
except ImportError:                                # 나눔고딕 파일이 이미 있는 컴퓨터
    import glob, os
    import matplotlib.font_manager as fm
    for f in glob.glob(os.path.expanduser("~/.fonts/Nanum*.ttf")):
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

def mae(a, b):                                      # 예측 채점: 평균 절대 오차 (식 12-5)
    return round(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))), 1)

URL_AIR = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/airline-passengers.csv")
air = pd.read_csv(URL_AIR, parse_dates=["Month"], index_col="Month")
y = air["Passengers"].asfreq("MS").astype(float)   # 매달 첫날(MS)을 시각으로
print(len(y), "개월:", y.index[0].strftime("%Y-%m"), "~", y.index[-1].strftime("%Y-%m"))

year_mean = y.groupby(y.index.year).mean()
print("1949년 평균", round(year_mean[1949], 1), "→ 1960년 평균", round(year_mean[1960], 1))
print("해마다 가장 붐빈 달:", y.groupby(y.index.year).idxmax().dt.month.value_counts().to_dict())

k = np.r_[0.5, np.ones(11), 0.5] / 12             # 무게 13개, 양 끝은 절반 (합은 1)
trend = y.rolling(13, center=True).apply(lambda w: w @ k)  # 창을 밀며 무게와 내적
print("무게의 합:", k.sum())
print("추세 1949-07:", round(trend["1949-07-01"], 1), " 1960-06:", round(trend["1960-06-01"], 1))

plt.figure(figsize=(7, 3))
plt.plot(y, color="#8EC5E8", label="승객 수")
plt.plot(trend, color="#E04B3C", label="12개월 이동평균(추세)")
plt.ylabel("승객(천 명)")
plt.legend()
plt.show()

# ### 12-2
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

dec = seasonal_decompose(y, model="multiplicative", period=12)  # 추세 × 계절 × 잔차
season = dec.seasonal.iloc[:12]
season.index = season.index.month
print("달별 계절 지수:", season.round(3).to_dict())
print("두 추세의 가장 큰 차이:", round((dec.trend - trend).abs().max(), 6))

ly = np.log(y)
steps = {"원자료": y, "로그": ly, "로그+1차 차분": ly.diff(),
         "로그+1차+계절 차분": ly.diff().diff(12)}
for name, s in steps.items():
    p = adfuller(s.dropna(), autolag="AIC")[1]
    print(f"{name:12s} ADF p값 {p:.4f}")

# ### 12-3
train, test = y.loc[:"1958"], y.loc["1959":]     # 1958년까지 배우고 1959~1960년(24달) 시험
h = len(test)
mean_f = np.repeat(train.mean(), h)                 # 지금까지의 평균
naive = np.repeat(train.iloc[-1], h)                # 마지막 값 그대로
slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
drift = train.iloc[-1] + slope * np.arange(1, h + 1)  # 첫 값과 끝 값을 이은 기울기로 연장
snaive = np.tile(train.iloc[-12:].values, 2)       # 작년 같은 달

base = {"평균": mean_f, "직전 값": naive, "추세(drift)": drift, "작년 같은 달": snaive}
for name, f in base.items():
    print(f"{name:10s} MAE {mae(test, f):6.1f} (천 명)")

pd.Series({k: mae(test, f) for k, f in base.items()}).plot.barh(
    color=["#8EC5E8", "#8EC5E8", "#8EC5E8", "#E04B3C"], figsize=(6, 2.6))
plt.xlabel("평균 오차 MAE(천 명)")
plt.show()

# ### 12-4
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def airline(series):                                # 박스와 젠킨스의 항공사 모델
    return SARIMAX(np.log(series), order=(0, 1, 1),
                   seasonal_order=(0, 1, 1, 12)).fit(disp=False)

def holt(series):                                   # 홀트-윈터스 (추세 + 곱셈 계절)
    return ExponentialSmoothing(series, trend="add", seasonal="mul",
                                seasonal_periods=12).fit()

sar = airline(train)
p_sar = np.exp(sar.forecast(h))                     # 로그로 배웠으니 지수로 되돌린다
p_hw = holt(train).forecast(h)
print("항공사 모델 θ =", round(sar.params["ma.L1"], 3), " Θ =", round(sar.params["ma.S.L12"], 3))
print("항공사 모델 MAE", mae(test, p_sar), "/ 홀트-윈터스 MAE", mae(test, p_hw))
print("1959년 1~3월 항공사 모델 예측:", p_sar.iloc[:3].round(1).tolist())

rows = []
for Y in range(1955, 1961):                         # 기준점을 한 해씩 옮긴다
    tr, te = y.loc[:str(Y - 1)], y.loc[str(Y)]
    f_sn = tr.iloc[-12:].values
    f_sar = np.exp(airline(tr).forecast(12)).values
    f_hw = holt(tr).forecast(12).values
    rows.append({"시험 해": Y, "작년 같은 달": mae(te, f_sn), "SARIMA": mae(te, f_sar),
                 "홀트-윈터스": mae(te, f_hw), "둘의 평균": mae(te, (f_sar + f_hw) / 2)})
roll = pd.DataFrame(rows).set_index("시험 해")
print(roll.to_string())
print("해마다 1등:", roll.iloc[:, :3].idxmin(axis=1).value_counts().to_dict())
print("여섯 해 평균:", roll.mean().round(1).to_dict())

fc = sar.get_forecast(h).conf_int(alpha=0.05)       # 1959~1960년의 95% 예측 구간
lo, hi = np.exp(fc.iloc[:, 0]).values, np.exp(fc.iloc[:, 1]).values
print("95% 구간 안에 든 달:", int(((test.values >= lo) & (test.values <= hi)).sum()), "/", h)
f61 = airline(y).get_forecast(12)                   # 1960년까지 모두 써서 1961년을 내다본다
mid, ci = np.exp(f61.predicted_mean), np.exp(f61.conf_int(alpha=0.05))
print("1961년 1월", round(mid.iloc[0], 1), ci.iloc[0].round(1).tolist(),
      "/ 7월", round(mid.iloc[6], 1), ci.iloc[6].round(1).tolist())

# ### 12-5
URL_EL = ("https://raw.githubusercontent.com/jenfly/opsd/"
          "master/opsd_germany_daily.csv")
el = pd.read_csv(URL_EL, parse_dates=["Date"], index_col="Date").asfreq("D")
c = el["Consumption"]                               # 하루 전력 소비, 단위 GWh
print(el.shape, el.index[0].date(), "~", el.index[-1].date())
print("빈칸:", el.isna().sum().to_dict())
print("가장 적은 날", c.idxmin().date(), round(c.min(), 1), "/ 가장 많은 날", c.idxmax().date(), round(c.max(), 1))

ym = c.groupby(c.index.year).mean()
change = (ym.pct_change() * 100).round(1)
print("전년 대비 변화(%):", change.loc[2009:2015].to_dict())
m = c.resample("MS").mean()                         # 달마다 평균
ratio = (m.loc["2014"].values / m.loc["2013"].values - 1) * 100
print("2014년 달마다 전년 같은 달 대비(%):", np.round(ratio, 1))

ym.plot.bar(color=["#E04B3C" if k == 2014 else "#8EC5E8" for k in ym.index], figsize=(7, 3))
plt.ylim(1200, 1420)
plt.ylabel("하루 평균 수요(GWh)")
plt.xlabel("")
plt.show()

# ### 12-6
dow = c.groupby(c.index.dayofweek).mean()          # 0=월요일 … 6=일요일
print("요일 평균(월~일):", dow.round(1).tolist())
print("주말은 평일보다", round((1 - dow[[5, 6]].mean() / dow[:5].mean()) * 100, 1), "% 적다")
mon = c.groupby(c.index.month).mean()
print("가장 많은 달", mon.idxmax(), "월", round(mon.max(), 1), "/ 가장 적은 달", mon.idxmin(), "월", round(mon.min(), 1))

lags = [1, 2, 3, 4, 5, 6, 7, 14, 364, 365]
r = pd.Series({k: c.autocorr(k) for k in lags})  # 시차 k만큼 민 자기 자신과의 상관계수
print("자기상관:", r.round(3).to_dict())

r.plot.bar(color=["#E04B3C" if k in (7, 14, 364) else "#8EC5E8" for k in lags], figsize=(7, 3))
plt.xlabel("시차(일)")
plt.ylabel("자기상관")
plt.xticks(rotation=0)
plt.show()

# ### 12-7
import logging
from prophet import Prophet
logging.getLogger("cmdstanpy").disabled = True    # 학습 진행 메시지를 숨긴다

tr_e, te_e = c.loc[:"2016"], c.loc["2017"]          # 2016년까지 배우고 2017년 365일 시험
f_mean = np.repeat(tr_e.loc["2016"].mean(), 365)
f_364 = np.tile(tr_e.iloc[-364:].values, 2)[:365]   # 52주 전 같은 요일 (2017년 값은 쓰지 않는다)
leak = c.shift(364).loc["2017"]                     # 흔한 실수: 시험 해의 값이 섞인다
print("shift(364)가 12월 31일 자리에 넣은 값의 날짜:", (te_e.index[-1] - pd.Timedelta(days=364)).date())
print("2016년 평균   MAE", mae(te_e, f_mean))
print("작년 같은 요일 MAE", mae(te_e, f_364), " (shift(364)로 하면", mae(te_e, leak), ")")

def fit_prophet(series, holidays_on, days=365):
    df = series.reset_index()
    df.columns = ["ds", "y"]                        # 프로펫은 열 이름이 ds, y여야 한다
    m = Prophet(daily_seasonality=False)
    if holidays_on:
        m.add_country_holidays(country_name="DE")   # 독일 공휴일 달력을 알려 준다
    m.fit(df)
    future = m.make_future_dataframe(periods=days)
    np.random.seed(0)                               # 예측 구간은 무작위 표본으로 만든다
    return m, m.predict(future).set_index("ds")

m0, fc0 = fit_prophet(tr_e, False)
m1, fc1 = fit_prophet(tr_e, True)
p0, p1 = fc0["yhat"].loc["2017"], fc1["yhat"].loc["2017"]
print("프로펫 기본   MAE", mae(te_e, p0))
print("프로펫 + 휴일 MAE", mae(te_e, p1))
err = (te_e - p1).abs().sort_values(ascending=False)
d0 = err.index[0]
print("가장 크게 틀린 날:", d0.date(), " 실제", round(te_e[d0], 1), " 예측", round(p1[d0], 1))
print("일주일 전 화요일", round(c["2017-10-24"], 1), "/ 10월 3일(통일의 날)", round(c["2017-10-03"], 1))
print("배운 휴일:", list(m1.train_holiday_names))
inside = ((te_e >= fc1["yhat_lower"].loc["2017"]) & (te_e <= fc1["yhat_upper"].loc["2017"])).mean()
print("80% 예측 구간 안에 든 날:", round(inside * 100, 1), "%")

plt.figure(figsize=(7, 3))
plt.plot(te_e.loc["2017-10"], color="black", label="실제")
plt.plot(p1.loc["2017-10"], color="#E04B3C", label="프로펫 + 휴일")
plt.plot(pd.Series(f_364, te_e.index).loc["2017-10"], color="#8EC5E8", label="작년 같은 요일")
plt.ylabel("GWh")
plt.legend()
plt.show()

# ### 12-8
rows = []
for Y in range(2012, 2018):                         # 2012~2017년을 차례로 시험
    tr, te = c.loc[:str(Y - 1)], c.loc[str(Y)]
    sn = np.tile(tr.iloc[-364:].values, 2)[:len(te)]  # 시험 해의 값은 쓰지 않는다
    p = fit_prophet(tr, True, len(te))[1]["yhat"].loc[str(Y)].values
    rows.append({"시험 해": Y, "작년 같은 요일": mae(te, sn), "프로펫+휴일": mae(te, p),
                 "둘의 평균": mae(te, (sn + p) / 2)})
eroll = pd.DataFrame(rows).set_index("시험 해")
print(eroll.to_string())
print("두 방법 가운데 1등:", eroll.iloc[:, :2].idxmin(axis=1).value_counts().to_dict())
print("여섯 해 평균:", eroll.mean().round(1).to_dict())

# ### 12-9
import holidays
from sklearn.ensemble import RandomForestRegressor

de = holidays.Germany(years=range(2006, 2018))      # 독일 전국 공휴일
hol = pd.Series([d in de for d in c.index.date], index=c.index).astype(int)
wk = c.index.dayofweek < 5                          # 월~금
print("평일 공휴일 평균", round(c[wk & (hol == 1)].mean(), 1), "/ 보통 평일 평균", round(c[wk & (hol == 0)].mean(), 1))
X = pd.DataFrame({"어제": c.shift(1), "지난주 같은 요일": c.shift(7),
                  "작년 같은 요일": c.shift(364), "지난 7일 평균": c.shift(1).rolling(7).mean(),
                  "달": c.index.month, "요일": c.index.dayofweek,
                  "휴일": hol, "어제가 휴일": hol.shift(1, fill_value=0)})
data = X.join(c.rename("y")).dropna()               # 앞의 364일은 작년 값이 없어 빠진다
trB, teB = data.loc[:"2016"], data.loc["2017"]
feats = [k for k in data.columns if k != "y"]
print("학습", len(trB), "일 / 시험", len(teB), "일")
print("어제 값 MAE", mae(teB.y, teB["어제"]), "/ 지난주 같은 요일 MAE", mae(teB.y, teB["지난주 같은 요일"]))

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf.fit(trB[feats], trB.y)
pred = rf.predict(teB[feats])
print("랜덤 포레스트 MAE", mae(teB.y, pred), f"(평균 수요의 {mae(teB.y, pred) / teB.y.mean() * 100:.1f}%)")
imp = pd.Series(rf.feature_importances_, feats).sort_values()
print(imp.sort_values(ascending=False).round(3).head(4).to_string())

imp.plot.barh(color="#E04B3C", figsize=(6, 3))
plt.xlabel("중요도")
plt.show()

# ### EX-6
no_hol = [k for k in feats if "휴일" not in k]       # 휴일 두 열을 뺀다
rf2 = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
rf2.fit(trB[no_hol], trB.y)
pred2 = rf2.predict(teB[no_hol])
is_hol = (teB["휴일"] == 1).values
print("한 해 전체 MAE — 휴일 있음", mae(teB.y, pred), "/ 없음", mae(teB.y, pred2))
print("휴일", is_hol.sum(), "일의 MAE — 휴일 있음", mae(teB.y[is_hol], pred[is_hol]),
      "/ 없음", mae(teB.y[is_hol], pred2[is_hol]))

# ### MP-0
# ☆ 공공데이터포털에서 받은 해마다의 파일을 하나로 합치기 (강의 자료의 파일을 쓰면 건너뜁니다)
import glob
RAW = sorted(glob.glob("한국전력거래소_시간별*전력수요량*.csv"))  # 받은 파일 이름에 맞게 고칩니다
if not RAW:
    print("포털에서 받은 파일이 없습니다. 강의 자료의 합친 파일로 1단계를 하십시오.")
else:
    parts = []
    for f in RAW:
        t = pd.read_csv(f, encoding="cp949")       # 포털 파일의 한글은 cp949
        t.columns = ["날짜"] + [f"{h}시" for h in range(1, 25)]  # 해마다 열 이름을 똑같이
        parts.append(t)
    kp_all = pd.concat(parts, ignore_index=True)    # 위아래로 잇는다
    kp_all = kp_all.dropna(subset=["날짜"])         # 내용 없는 줄을 버린다
    kp_all["날짜"] = pd.to_datetime(kp_all["날짜"])
    kp_all = kp_all.sort_values("날짜").drop_duplicates("날짜", keep="last")
    kp_all.to_csv("kpx_직접_합친_파일.csv", index=False)  # 1단계의 PATH를 이 이름으로
    print(len(RAW), "개 파일 →", kp_all.shape, kp_all["날짜"].min().date(), "~", kp_all["날짜"].max().date())

# ### MP-1
# 데이터 바꾸기: 다른 해나 다른 파일을 쓸 때는 이 세 줄만 고칩니다.
PATH = "kpx_시간별_전국_전력수요량_2021_2025.csv"   # Colab 왼쪽 파일 창에 올린 파일
DATE = "날짜"                                       # 날짜 열 (나머지 24열은 1시~24시의 수요, MW)
TEST = "2025"                                       # 시험할 해 (그 앞은 모두 학습에 쓴다)

kp = pd.read_csv(PATH)
print(kp.shape, "/ 빈칸", kp.isna().sum().sum(), "/ 날짜가 빈 줄", kp[DATE].isna().sum())
kp = kp.dropna(subset=[DATE])                       # 내용 없는 줄을 버린다
kp[DATE] = pd.to_datetime(kp[DATE])
print("같은 날이 두 번:", kp[DATE].duplicated().sum(), "/ 날짜 순서대로인가:", kp[DATE].is_monotonic_increasing)
kp = kp.set_index(DATE).sort_index()
day = (kp.mean(axis=1) * 24 / 1000).asfreq("D")     # 하루 GWh = 24시간 평균(MW) × 24 ÷ 1000
print(len(day), "일, 빈칸", day.isna().sum())
print("가장 적은 날", day.idxmin().date(), round(day.min(), 1), "/ 가장 많은 날", day.idxmax().date(), round(day.max(), 1))

# ### MP-2
for k in [1, 7, 364]:
    print(f"시차 {k:3d}일 자기상관 {day.autocorr(k):.3f}")
tY = day.loc[TEST]
print(TEST + "년 다음 날 예측 — 어제 값 MAE", mae(tY, day.shift(1).loc[TEST]),
      "/ 지난주 같은 요일 MAE", mae(tY, day.shift(7).loc[TEST]))
print("달별 평균(GWh):", day.groupby(day.index.month).mean().round(0).astype(int).tolist())

# ### MP-3
import holidays
from sklearn.ensemble import RandomForestRegressor
kr = holidays.KR(years=range(day.index.year[0], day.index.year[-1] + 1))  # 설·추석 연휴, 대체 휴일까지
hol_k = pd.Series([d in kr for d in day.index.date], index=day.index).astype(int)
Xk = pd.DataFrame({"어제": day.shift(1), "지난주 같은 요일": day.shift(7),
                   "작년 같은 요일": day.shift(364), "지난 7일 평균": day.shift(1).rolling(7).mean(),
                   "달": day.index.month, "요일": day.index.dayofweek,
                   "휴일": hol_k, "어제가 휴일": hol_k.shift(1, fill_value=0)})
dk = Xk.join(day.rename("y")).dropna()
trK, teK = dk.loc[:str(int(TEST) - 1)], dk.loc[TEST]
fk = list(Xk.columns)
rfk = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
rfk.fit(trK[fk], trK.y)
pk = rfk.predict(teK[fk])
print("학습", len(trK), "일 / 시험", len(teK), "일")
print("랜덤 포레스트 MAE", mae(teK.y, pk), f"(평균 수요의 {mae(teK.y, pk) / teK.y.mean() * 100:.1f}%)")
imp_k = pd.Series(rfk.feature_importances_, fk).sort_values(ascending=False)
print(imp_k.round(3).head(3).to_string())

# ### MP-4
e = pd.Series(np.abs(teK.y.values - pk), index=teK.index)
print("달별 MAE:", e.groupby(e.index.month).mean().round(1).tolist())
big = pd.DataFrame({"실제": teK.y, "예측": pk}).loc[e.nlargest(5).index].round(1)  # 가장 크게 틀린 닷새
big["공휴일"] = [kr.get(d, "") for d in big.index.date]
print(big.to_string())

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
