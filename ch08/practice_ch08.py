# 08장. 내일의 발전량: 회귀 분석 (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 8-k' 조각 하나가 책의 [파이썬 실습 8-k] 하나입니다. MP-k는 [미니 프로젝트 8]의 단계, EX-k는 연습문제 정답 코드입니다.
# 한국 데이터 세 파일(한국전력거래소 태양광·전력수요, 기상청 ASOS 서울)은 코랩 왼쪽 파일 창에 올려 두고 시작합니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: pandas, numpy, statsmodels, scikit-learn, matplotlib, holidays)

# ### 8-1
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
DATA = ""      # 수업 자료로 받은 한국 데이터 CSV를 둔 폴더 (코랩 파일 창에 바로 올렸으면 빈칸 그대로)

URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
       "master/csv/HistData/GaltonFamilies.csv")
galton = pd.read_csv(URL)
sons = galton[galton["gender"] == "male"].copy()
sons["아버지"] = sons["father"] * 2.54            # 인치 → cm
sons["아들"] = sons["childHeight"] * 2.54
print("자녀", len(galton), "명 가운데 아들", len(sons), "명")
print(sons[["아버지", "아들"]].agg(["mean", "std"]).round(2).to_string())

tall = sons[sons["아버지"] >= sons["아버지"].quantile(0.9)]     # 키 큰 아버지 상위 10%
short = sons[sons["아버지"] <= sons["아버지"].quantile(0.1)]    # 키 작은 아버지 하위 10%
tall_son = sons[sons["아들"] >= sons["아들"].quantile(0.9)]    # 거꾸로: 키 큰 아들 상위 10%
print(f"키 큰 아버지 {len(tall)}명: {tall['아버지'].mean():.1f}cm → 아들 {tall['아들'].mean():.1f}cm")
print(f"키 작은 아버지 {len(short)}명: {short['아버지'].mean():.1f}cm → 아들 {short['아들'].mean():.1f}cm")
print(f"키 큰 아들 {len(tall_son)}명: {tall_son['아들'].mean():.1f}cm → 그 아버지 {tall_son['아버지'].mean():.1f}cm")

b, a = np.polyfit(sons["아버지"], sons["아들"], 1)   # 기울기, 절편
r = sons["아버지"].corr(sons["아들"])
print(f"회귀선: 아들 = {a:.1f} + {b:.3f} × 아버지,  상관계수 {r:.3f},  R² {r ** 2:.3f}")

plt.figure(figsize=(5, 4.4))
plt.scatter(sons["아버지"], sons["아들"], s=8, alpha=0.4, color="gray")
xs = np.array([155, 200])
plt.plot(xs, a + b * xs, color="#C8234B", label="회귀선")
plt.plot(xs, xs, "k--", lw=1, label="아들 = 아버지")
plt.xlabel("아버지 키(cm)")
plt.ylabel("아들 키(cm)")
plt.legend()
plt.show()

# ### 8-2
study = pd.DataFrame({"공부시간": [2, 4, 6, 8, 10], "점수": [60, 70, 75, 85, 90]})
x, y = study["공부시간"], study["점수"]
sxy = ((x - x.mean()) * (y - y.mean())).sum()     # 편차 곱의 합
sxx = ((x - x.mean()) ** 2).sum()                 # x 편차 제곱의 합
w = sxy / sxx
b = y.mean() - w * x.mean()
print(f"기울기 w = {sxy} ÷ {sxx} = {w},  절편 b = {b}")

pred = b + w * x
sse = ((y - pred) ** 2).sum()
sst = ((y - y.mean()) ** 2).sum()
print("잔차:", (y - pred).tolist(), " 잔차의 합:", (y - pred).sum())
print(f"SSE = {sse},  SST = {sst},  R² = {1 - sse / sst:.3f}")

ws = np.linspace(2.5, 5.0, 51)                    # 기울기를 바꿔 가며 SSE 재기
sses = [((y - (y.mean() - v * x.mean() + v * x)) ** 2).sum() for v in ws]
for v in [3.0, 3.5, 3.75, 4.0, 4.5]:
    print(f"w = {v:4} → SSE = {((y - (y.mean() - v * x.mean() + v * x)) ** 2).sum():5.1f}")
plt.figure(figsize=(5, 3.2))
plt.plot(ws, sses, color="gray")
plt.scatter([w], [sse], color="#C8234B", zorder=3)
plt.xlabel("기울기 w")
plt.ylabel("잔차 제곱합(SSE)")
plt.show()

# ### 8-3
xa, ya = x.to_numpy(float), y.to_numpy(float)
w_gd, b_gd, lr = 0.0, 0.0, 0.01                   # 0에서 출발, 한 걸음 크기(학습률) 0.01
losses = []
for step in range(3000):
    err = (b_gd + w_gd * xa) - ya                 # 예측 − 실제
    losses.append((err ** 2).mean())
    if step in (0, 100, 1000):
        print(f"{step:4d}걸음: w = {w_gd:6.3f}, b = {b_gd:6.3f}, 손실 = {losses[-1]:8.2f}")
    w_gd -= lr * 2 * (err * xa).mean()            # 손실이 줄어드는 쪽으로 w 이동
    b_gd -= lr * 2 * err.mean()                   # 손실이 줄어드는 쪽으로 b 이동
print(f"3000걸음: w = {w_gd:.3f}, b = {b_gd:.3f}, 손실 = {losses[-1]:.2f}")

plt.figure(figsize=(5, 3.2))
plt.plot(losses, color="#C8234B")
plt.yscale("log")
plt.xlabel("걸음 수")
plt.ylabel("손실(로그 눈금)")
plt.show()

# ### 8-4
SOLAR = "kpx_지역별_시간별_태양광풍력_발전량_2025.csv"   # 수업 자료로 받은 정리본
ASOS = "asos_서울108_시간자료_전체열_2023_2026.csv"

kpx = pd.read_csv(DATA + SOLAR)
print("한국전력거래소 파일:", kpx.shape, "/ 지역", kpx["지역"].nunique(), "곳")
seoul = kpx[(kpx["지역"] == "서울시") & (kpx["연료원"] == "태양광")].copy()
seoul["일시"] = pd.to_datetime(seoul["거래일"]) + pd.to_timedelta(seoul["거래시간"], unit="h")
seoul = seoul.rename(columns={"전력거래량(MWh)": "발전량"})[["일시", "발전량"]]
print(f"서울 태양광: {len(seoul)}시각, 한 해 {seoul['발전량'].sum():,.0f}MWh, 가장 많은 시각 {seoul['발전량'].max():.1f}MWh")

names = {"일시": "일시", "기온(°C)": "기온", "습도(%)": "습도", "일사(MJ/m2)": "일사",
         "전운량(10분위)": "운량", "적설(cm)": "적설"}
wx = pd.read_csv(DATA + ASOS, usecols=list(names)).rename(columns=names)
wx["일시"] = pd.to_datetime(wx["일시"])
print("기상청 서울 관측:", wx.shape, wx["일시"].min(), "~", wx["일시"].max())

for h in [-1, 0, 1]:                               # 시각을 한 칸씩 밀어 보며 잇기
    moved = wx[["일시", "일사"]].assign(일시=wx["일시"] + pd.Timedelta(hours=h))
    m = seoul.merge(moved, on="일시")
    print(f"기상 시각을 {h:+d}시간 밀면 상관계수 {m['발전량'].corr(m['일사'].fillna(0)):.3f}")

df = seoul.merge(wx, on="일시", how="left")        # 같은 시각의 날씨를 옆에 붙인다
df["시각"] = df["일시"].dt.hour
blank = df["일사"].isna()
noon = blank & df["시각"].between(8, 18)
print("일사 빈칸", blank.sum(), "시각 / 그 가운데 한낮(8~18시)", noon.sum(), "시각:",
      df.loc[noon, "일시"].dt.date.unique())
night = blank & ~noon
print(f"밤인데 발전량이 0보다 큰 시각 {(df.loc[night, '발전량'] > 0).sum()}개, 합계 {df.loc[night, '발전량'].sum():.1f}MWh")

df.groupby("시각")["발전량"].mean().plot.bar(color="gray", figsize=(6, 3))
plt.ylabel("평균 발전량(MWh)")
plt.xlabel("시각(그 시각까지 한 시간)")
plt.xticks(range(0, 24, 3), range(0, 24, 3), rotation=0)
plt.show()

# ### 8-5
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

day = df[df["일사"] > 0].copy()                    # 해가 있는 시각만
print("해가 있는 시각:", len(day))
m1 = smf.ols("발전량 ~ 일사", data=day).fit()
print(f"발전량 = {m1.params['Intercept']:.2f} + {m1.params['일사']:.2f} × 일사,  R² {m1.rsquared:.3f}")
print("밤까지 넣으면 R²", round(smf.ols("발전량 ~ 일사", data=df.fillna({"일사": 0})).fit().rsquared, 3))

CUT = "2025-11-01"                                 # 앞 10달로 배우고 뒤 2달로 시험
train, test = day[day["일시"] < CUT], day[day["일시"] >= CUT]
print("학습", len(train), "시각 / 시험", len(test), "시각")

def score(name, y_true, y_pred, unit="MWh"):     # 같은 채점표로 모델을 견준다
    print(f"{name:10s} MAE {mean_absolute_error(y_true, y_pred):6.2f}{unit}   R² {r2_score(y_true, y_pred):6.3f}")

score("평균으로 찍기", test["발전량"], np.full(len(test), train["발전량"].mean()))
score("시각별 평균", test["발전량"], test["시각"].map(train.groupby("시각")["발전량"].mean()))
yday = df.set_index("일시")["발전량"].shift(24)     # 24시각 전 = 어제 같은 시각
score("어제 같은 시각", test["발전량"], test["일시"].map(yday))
fit = smf.ols("발전량 ~ 일사", data=train).fit()
score("일사 회귀", test["발전량"], fit.predict(test))
print(f"  시험 잔차(실제 − 예측)의 평균 {(test['발전량'] - fit.predict(test)).mean():+.2f}MWh")
r_tr, r_te = train_test_split(day, test_size=0.2, random_state=42)
score("(랜덤 분할)", r_te["발전량"], smf.ols("발전량 ~ 일사", data=r_tr).fit().predict(r_te))

plt.figure(figsize=(5, 4))
plt.scatter(day["일사"], day["발전량"], s=4, alpha=0.3, color="gray")
xs = np.array([0, 3.9])
plt.plot(xs, m1.params["Intercept"] + m1.params["일사"] * xs, color="#C8234B")
plt.xlabel("일사(MJ/m², 한 시간)")
plt.ylabel("서울 태양광 발전량(MWh)")
plt.show()

# ### 8-6
day["월"] = day["일시"].dt.month
slope = {mo: smf.ols("발전량 ~ 일사", data=g).fit().params["일사"] for mo, g in day.groupby("월")}
print("달마다 다시 그은 직선의 기울기:")
print(pd.Series(slope).round(2).to_string())

day["온도손실"] = day["일사"] * (day["기온"] - 25)  # 햇빛이 셀 때 25℃보다 더운 만큼
train, test = day[day["일시"] < CUT], day[day["일시"] >= CUT]
forms = {"M1": "발전량 ~ 일사",
         "M2": "발전량 ~ 일사 + 기온 + 운량 + 습도",
         "M3": "발전량 ~ 일사 + 기온 + 운량 + 습도 + 온도손실",
         "M4": "발전량 ~ 일사 + 기온 + 운량 + 습도 + 온도손실 + C(시각)"}
for name, f in forms.items():
    fit = smf.ols(f, data=train).fit()
    tr_mae = mean_absolute_error(train["발전량"], fit.fittedvalues)
    print(f"{name} 학습 MAE {tr_mae:.2f}, R² {fit.rsquared:.3f} | ", end="")
    score("시험", test["발전량"], fit.predict(test))
m3 = smf.ols(forms["M3"], data=train).fit()
print(f"온도손실 계수 {m3.params['온도손실']:.4f} (p값 {m3.pvalues['온도손실']:.1e})")
print(f"기온 1℃마다 일사 1당 발전량이 {-m3.params['온도손실'] / m3.params['일사']:.1%}씩 준다")

plt.figure(figsize=(6, 3))
plt.bar(list(slope), list(slope.values()), color="gray")
plt.xticks(range(1, 13))
plt.xlabel("월")
plt.ylabel("일사 1당 발전량(MWh)")
plt.show()

# ### 8-7
from statsmodels.stats.stattools import durbin_watson

cond = pd.DataFrame({"일사": [2.0], "기온": [10.0], "운량": [2], "습도": [50]})
cond["온도손실"] = cond["일사"] * (cond["기온"] - 25)
pf = m3.get_prediction(cond).summary_frame(alpha=0.05)
print(pf[["mean", "mean_ci_lower", "mean_ci_upper", "obs_ci_lower", "obs_ci_upper"]].to_string(index=False, float_format="%.2f"))

band = m3.get_prediction(test).summary_frame(alpha=0.05)
inside = (test["발전량"] >= band["obs_ci_lower"]) & (test["발전량"] <= band["obs_ci_upper"])
print(f"시험 {len(test)}시각 가운데 95% 예측구간 안: {inside.mean():.1%}"
      f" (아래로 벗어남 {(test['발전량'] < band['obs_ci_lower']).sum()}, 위로 {(test['발전량'] > band['obs_ci_upper']).sum()})")

res = test["발전량"] - m3.predict(test)
print("잔차 가운데 값", round(res.median(), 2), "/ 가장 크게 틀린 시각:")
worst = test.assign(잔차=res).nsmallest(4, "잔차")
print(worst[["일시", "발전량", "일사", "기온", "적설", "잔차"]].to_string(index=False, float_format="%.2f"))
print("더빈-왓슨(M3 학습 잔차):", round(durbin_watson(m3.resid), 2))

plt.figure(figsize=(5, 3.2))
plt.hist(res, bins=40, color="gray")
plt.axvline(0, color="#C8234B")
plt.xlabel("잔차 = 실제 - 예측(MWh)")
plt.ylabel("시각 수")
plt.show()

# ### 8-8
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

dew = pd.read_csv(DATA + ASOS, usecols=["일시", "이슬점온도(°C)"]).rename(columns={"이슬점온도(°C)": "이슬점"})
dew["일시"] = pd.to_datetime(dew["일시"])
d8 = day.merge(dew, on="일시")
for cols in [["일사", "기온", "습도"], ["일사", "기온", "습도", "이슬점"]]:
    X = sm.add_constant(d8[cols])
    vif = [variance_inflation_factor(X.values, i) for i in range(1, len(cols) + 1)]
    print(" + ".join(cols), "→ VIF", ", ".join(f"{v:.1f}" for v in vif))
m_3 = smf.ols("발전량 ~ 일사 + 기온 + 습도", data=d8).fit()
m_dew = smf.ols("발전량 ~ 일사 + 기온 + 습도 + 이슬점", data=d8).fit()
print(pd.DataFrame({"이슬점 없이": m_3.params, "이슬점 넣고": m_dew.params}).round(3).to_string())
print("R²:", round(m_3.rsquared, 4), "→", round(m_dew.rsquared, 4))

# ### 8-9
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor

feats = ["일사", "기온", "운량", "습도", "온도손실", "시각"]
train, test = day[day["일시"] < CUT], day[day["일시"] >= CUT]
models = {"선형": make_pipeline(StandardScaler(), LinearRegression()),
          "릿지": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))),
          "라소": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0))}
for name, mdl in models.items():
    mdl.fit(train[feats], train["발전량"])
    score(name, test["발전량"], mdl.predict(test[feats]))
print("라소의 계수:", pd.Series(models["라소"][-1].coef_, index=feats).round(3).to_dict())
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
rf.fit(train[feats], train["발전량"])
score("랜덤 포레스트", test["발전량"], rf.predict(test[feats]))
score("  (학습 성적)", train["발전량"], rf.predict(train[feats]))

# ### MP-1
DEMAND = "kpx_시간별_전국_전력수요량_2021_2025.csv"  # 수업 자료로 받은 정리본
dem = pd.read_csv(DATA + DEMAND)
print(dem.shape, "/ 날짜가 빈 줄", dem["날짜"].isna().sum())
dem = dem.dropna(subset=["날짜"])
dem["날짜"] = pd.to_datetime(dem["날짜"])
dem["수요"] = dem.drop(columns="날짜").sum(axis=1) / 1000   # 24시간 MW의 합 → 하루 GWh

temp = pd.read_csv(DATA + ASOS, usecols=["일시", "기온(°C)"])
temp["날짜"] = pd.to_datetime(temp["일시"]).dt.normalize()
temp = temp.groupby("날짜")["기온(°C)"].mean().rename("기온").reset_index()

mp = dem[["날짜", "수요"]].merge(temp, on="날짜")       # 두 자료가 겹치는 날만
print(len(mp), "일:", mp["날짜"].min().date(), "~", mp["날짜"].max().date())
print(mp[["수요", "기온"]].describe().round(1).loc[["mean", "min", "max"]].to_string())

# ### MP-2
mp["연도"] = mp["날짜"].dt.year
tr, te = mp[mp["연도"] < 2025], mp[mp["연도"] == 2025]   # 2023~2024년으로 배우고 2025년으로 시험
line = smf.ols("수요 ~ 기온", data=tr).fit()
print(f"직선: 수요 = {line.params['Intercept']:.1f} {line.params['기온']:+.2f} × 기온,  학습 R² {line.rsquared:.3f}")
print(f"상관계수 {mp['수요'].corr(mp['기온']):.3f}")
score("평균으로 찍기", te["수요"], np.full(len(te), tr["수요"].mean()), "GWh")
score("직선", te["수요"], line.predict(te), "GWh")

plt.figure(figsize=(5, 3.6))
plt.scatter(mp["기온"], mp["수요"], s=6, alpha=0.4, color="gray")
plt.xlabel("서울 하루 평균 기온(°C)")
plt.ylabel("전국 하루 전력 수요(GWh)")
plt.show()

# ### MP-3
import holidays
kr = holidays.KR(years=[2023, 2024, 2025])
mp["쉬는날"] = ((mp["날짜"].dt.dayofweek >= 5) | mp["날짜"].isin(pd.to_datetime(list(kr)))).astype(int)
tr, te = mp[mp["연도"] < 2025], mp[mp["연도"] == 2025]
u = smf.ols("수요 ~ 기온 + I(기온 ** 2)", data=tr).fit()
u2 = smf.ols("수요 ~ 기온 + I(기온 ** 2) + 쉬는날", data=tr).fit()
print(f"U자: 학습 R² {u.rsquared:.3f}, 수요가 가장 적은 기온 {-u.params['기온'] / (2 * u.params['I(기온 ** 2)']):.1f}℃")
print(f"U자 + 쉬는날: 학습 R² {u2.rsquared:.3f}, 쉬는 날 계수 {u2.params['쉬는날']:.1f}GWh")
score("U자", te["수요"], u.predict(te), "GWh")
score("U자 + 쉬는날", te["수요"], u2.predict(te), "GWh")

t = np.linspace(mp["기온"].min(), mp["기온"].max(), 100)
plt.figure(figsize=(5, 3.6))
plt.scatter(tr["기온"], tr["수요"], s=6, alpha=0.4, color="gray")
plt.plot(t, u.predict(pd.DataFrame({"기온": t})), color="#C8234B")
plt.xlabel("서울 하루 평균 기온(°C)")
plt.ylabel("전국 하루 전력 수요(GWh)")
plt.show()

# ### MP-4
te = te.assign(잔차=te["수요"] - u2.predict(te))
print(te.nsmallest(5, "잔차")[["날짜", "수요", "기온", "쉬는날", "잔차"]].to_string(index=False, float_format="%.1f"))

# ### EX-6
CUT2 = "2025-07-01"                                # 앞 6달로 배우고 뒤 6달로 시험
train2, test2 = day[day["일시"] < CUT2], day[day["일시"] >= CUT2]
for name in ["M1", "M3"]:
    fit = smf.ols(forms[name], data=train2).fit()
    score(name, test2["발전량"], fit.predict(test2))

# ### RAW-1
# ☆ 원자료에서 정리본 만들기: 공공데이터포털·기상자료개방포털에서 받은 파일이 DATA 폴더에 있을 때만 돈다.
import glob
raw_asos = sorted(glob.glob(DATA + "OBS_ASOS_TIM_*.csv"))      # 기상자료개방포털: 한 번에 1년쯤씩 받은 파일들
if raw_asos:
    parts = [pd.read_csv(f, encoding="cp949") for f in raw_asos]  # 원자료는 한글 윈도 인코딩(cp949)
    asos = pd.concat(parts).drop_duplicates("일시").sort_values("일시")
    asos.to_csv(DATA + "asos_내가정리한_서울108.csv", index=False)  # UTF-8로 저장
    print("ASOS 파일", len(raw_asos), "개 →", len(asos), "시각", asos["일시"].iloc[0], "~", asos["일시"].iloc[-1])
    print("일사 빈칸(밤 포함):", asos["일사(MJ/m2)"].isna().sum(), "/ 강수량 빈칸(비가 안 온 시각):", asos["강수량(mm)"].isna().sum())
else:
    print("기상청 원자료(OBS_ASOS_TIM_*.csv)가 없어 건너뜁니다.")

raw_kpx = sorted(glob.glob(DATA + "한국전력거래소_*태양광*풍력*.csv"))  # 공공데이터포털에서 받은 파일
if raw_kpx:
    sol = pd.read_csv(raw_kpx[0], encoding="cp949")
    sol.columns = ["거래일", "거래시간", "지역", "연료원", "전력거래량(MWh)"]
    sol.to_csv(DATA + "kpx_내가정리한_태양광풍력.csv", index=False)
    print("한국전력거래소 파일:", sol.shape, "/ 거래시간", sol["거래시간"].min(), "~", sol["거래시간"].max())
else:
    print("한국전력거래소 원자료(한국전력거래소_*태양광*풍력*.csv)가 없어 건너뜁니다.")

raw_dem = sorted(glob.glob(DATA + "한국전력거래소_시간별*전력수요량*.csv"))  # 미니 프로젝트: 해마다 한 파일
if raw_dem:
    parts = []
    for f in raw_dem:
        t = pd.read_csv(f, encoding="cp949")
        t.columns = ["날짜"] + [f"{h}시" for h in range(1, 25)]   # 해마다 열 이름을 똑같이
        parts.append(t)
    dem_all = pd.concat(parts).dropna(subset=["날짜"])            # 내용 없는 줄을 버린다
    dem_all = dem_all.drop_duplicates("날짜").sort_values("날짜")
    dem_all.to_csv(DATA + "kpx_내가정리한_전력수요.csv", index=False)
    print("전력수요 파일", len(raw_dem), "개 →", dem_all.shape, dem_all["날짜"].iloc[0], "~", dem_all["날짜"].iloc[-1])
else:
    print("한국전력거래소 원자료(한국전력거래소_시간별*전력수요량*.csv)가 없어 건너뜁니다.")
