# 09장. 스팸일까, 아닐까: 분류 (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 9-k' 조각 하나가 책의 [파이썬 실습 9-k] 하나입니다. MP-k는 [미니 프로젝트 9]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v3.py  (필요 라이브러리: pandas, numpy, scikit-learn, matplotlib)
# 미니 프로젝트는 강의 자료의 두 파일(에어코리아 서울 도시대기 평균, 기상청 서울 ASOS)이 같은 폴더에 있어야 합니다.

# ### 9-1
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

URL = ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
       "master/data/sms.tsv")
sms = pd.read_csv(URL, sep="\t", header=None, names=["label", "msg"], quoting=3)
print("처음 읽은 문자:", len(sms), "통,  똑같은 문자:", sms.duplicated().sum(), "통")
sms = sms.drop_duplicates().reset_index(drop=True)
print(sms["label"].value_counts().to_string())
print("스팸 비율:", round((sms["label"] == "spam").mean(), 3))
print("모두 정상이라고 찍을 때 정확도:", round((sms["label"] == "ham").mean(), 3))

sms["스팸"] = (sms["label"] == "spam").astype(int)  # 스팸이면 1, 정상이면 0
sms["숫자"] = sms["msg"].str.count(r"\d")          # 문자 속 숫자(0~9)의 개수
print(sms.groupby("label")["숫자"].mean().round(1).to_string())
print("평균 글자 수:", sms["msg"].str.len().groupby(sms["label"]).mean().round(1).to_dict())

sms["label"].value_counts().plot.bar(color=["#8EC5E8", "#E04B3C"], figsize=(4, 3))
plt.title("정상 문자와 스팸 문자의 수")
plt.xticks(rotation=0)
plt.show()

# ### 9-2
bins = [-1, 0, 2, 5, 10, 20, 100]
group = pd.cut(sms["숫자"], bins)
table = sms.groupby(group, observed=True)["스팸"].agg(["size", "mean"])
print(table.round(3).to_string())
rate = table["mean"]

rate.plot.bar(color="#E04B3C", figsize=(6, 3))
plt.ylabel("스팸 비율")
plt.xlabel("문자 속 숫자 개수")
plt.xticks(rotation=0)
plt.show()

# ### 9-3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train, test = train_test_split(sms, test_size=0.25, random_state=42,
                               stratify=sms["스팸"])          # 75%로 배우고 25%로 시험
print("훈련", len(train), "통 / 시험", len(test), "통")

one = LogisticRegression().fit(train[["숫자"]], train["스팸"])
b0, b1 = one.intercept_[0], one.coef_[0][0]
print(f"절편 {b0:.2f}, 기울기 {b1:.3f}")
print("숫자 하나가 늘 때 오즈는", round(np.exp(b1), 2), "배")
print("확률이 0.5가 되는 숫자 개수:", round(-b0 / b1, 1))

x = np.arange(0, 21)
p = 1 / (1 + np.exp(-(b0 + b1 * x)))          # 시그모이드
for k in [0, 2, 5, 10, 15]:
    print(f"숫자 {k:2d}개 → 스팸일 확률 {p[k]:.3f}")

plt.figure(figsize=(6, 3))
plt.plot(x, p, color="#E04B3C")
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("문자 속 숫자 개수")
plt.ylabel("스팸일 확률")
plt.show()

# ### 9-4
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

pred = one.predict(test[["숫자"]])            # 문턱값 0.5로 판정
y = test["스팸"]
print(confusion_matrix(y, pred))             # [[TN FP] [FN TP]]
print("정확도:", round(accuracy_score(y, pred), 3))
print("정밀도:", round(precision_score(y, pred), 3))
print("재현율:", round(recall_score(y, pred), 3))
print("F1    :", round(f1_score(y, pred), 3))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["정상", "스팸"],
                                        cmap="Blues", colorbar=False)
plt.xlabel("모델의 판정")
plt.ylabel("실제")
plt.show()

# ### 9-5
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

low = sms["msg"].str.lower()                    # 단어가 든 비율은 문자 전체에서 센다
for w in ["free", "call"]:
    has = low.str.contains(rf"\b{w}\b")
    print(f'"{w}" 단어가 든 비율: 스팸 {has[sms["스팸"] == 1].mean():.1%}, 정상 {has[sms["스팸"] == 0].mean():.1%}')
both = low.str.contains(r"\bfree\b") & low.str.contains(r"\bcall\b")
print("두 단어가 모두 든 문자:", both.sum(), "통, 그 가운데 스팸", f'{sms.loc[both, "스팸"].mean():.1%}')

vec = CountVectorizer()                       # 단어마다 몇 번 나왔는지 센다
A_tr = vec.fit_transform(train["msg"])
A_te = vec.transform(test["msg"])
print("단어 종류:", A_tr.shape[1])
i = vec.vocabulary_["prize"]
is_spam = (train["스팸"] == 1).values
print('훈련 문자의 "prize": 스팸에', A_tr[is_spam][:, i].sum(), "번, 정상에", A_tr[~is_spam][:, i].sum(), "번")

nb = MultinomialNB().fit(A_tr, train["스팸"])
lr = LogisticRegression(max_iter=2000).fit(A_tr, train["스팸"])

rows = []
for name, m, X in [("숫자 개수 하나", one, test[["숫자"]]),
                   ("나이브 베이즈(단어)", nb, A_te),
                   ("로지스틱 회귀(단어)", lr, A_te)]:
    pr = m.predict(X)
    rows.append([name, accuracy_score(y, pr), precision_score(y, pr),
                 recall_score(y, pr), f1_score(y, pr)])
result = pd.DataFrame(rows, columns=["모델", "정확도", "정밀도", "재현율", "F1"])
print(result.round(3).to_string(index=False))

result.set_index("모델")[["정확도", "재현율"]].plot.barh(figsize=(6, 3),
                                                    color=["#8EC5E8", "#E04B3C"])
plt.axvline(0.874, color="gray", linestyle="--")   # 모두 정상이라고 찍을 때
plt.xlim(0, 1)
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.show()

# ### 9-6
prob = lr.predict_proba(A_te)[:, 1]           # 스팸일 확률
rows = []
for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
    tn, fp, fn, tp = confusion_matrix(y, (prob >= th).astype(int)).ravel()
    rows.append([th, fp, fn, 10 * fp + fn])
cost = pd.DataFrame(rows, columns=["문턱값", "헛경보", "놓침", "손해(헛경보×10+놓침)"])
print(cost.to_string(index=False))

cost.plot(x="문턱값", y=["헛경보", "놓침"], marker="o", figsize=(6, 3),
          color=["#3F7CC0", "#E04B3C"])
plt.ylabel("문자 수")
plt.show()

# ### 9-7
words = vec.get_feature_names_out()
coef = lr.coef_[0]
order = np.argsort(coef)
print("정상 쪽 단어:", list(words[order[:8]]))
print("스팸 쪽 단어:", list(words[order[-8:]][::-1]))

low = sms["msg"].str.lower()
for w in ["lor", "da", "&lt;#&gt;"]:
    has = low.str.contains(w, regex=False) if w.startswith("&") else low.str.contains(rf"\b{w}\b")
    print(f"'{w}' 든 문자: 스팸 {has[sms['스팸'] == 1].sum()}통, 정상 {has[sms['스팸'] == 0].sum()}통")

top = pd.Series(coef[order[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]]],
                index=words[order[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]]])
top.plot.barh(color=["#3F7CC0"] * 5 + ["#E04B3C"] * 5, figsize=(5, 4))
plt.xlabel("계수 (오른쪽일수록 스팸 쪽)")
plt.show()

# ### 9-8
rows = []
for n in [50, 200, 1000]:
    f_nb, f_lr = [], []
    for seed in range(10):                    # 무작위로 n통을 10번 뽑아 평균
        sub = train.sample(n, random_state=seed)
        v = CountVectorizer().fit(sub["msg"])
        a, b = v.transform(sub["msg"]), v.transform(test["msg"])
        f_nb.append(f1_score(y, MultinomialNB().fit(a, sub["스팸"]).predict(b)))
        f_lr.append(f1_score(y, LogisticRegression(max_iter=2000).fit(a, sub["스팸"]).predict(b)))
    rows.append([n, np.mean(f_nb), np.mean(f_lr)])
small = pd.DataFrame(rows, columns=["훈련 문자 수", "나이브 베이즈 F1", "로지스틱 F1"])
print(small.round(3).to_string(index=False))

# ### 9-9
from sklearn.metrics import roc_curve, roc_auc_score

for name, pr in [("숫자 개수 하나", one.predict_proba(test[["숫자"]])[:, 1]),
                 ("로지스틱 회귀(단어)", prob)]:
    fpr, tpr, _ = roc_curve(y, pr)
    plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, pr):.3f})")
    print(name, "AUC:", round(roc_auc_score(y, pr), 3))
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("헛경보 비율")
plt.ylabel("재현율")
plt.legend()
plt.show()

# ### MP-0
# ☆ 에어코리아에서 받은 달별 원자료를 서울 도시대기 평균 한 파일로 줄이기 (강의 자료의 파일을 쓰면 건너뜁니다)
import glob
RAW = sorted(glob.glob("20*년 *월.xlsx"))          # 받은 파일 이름에 맞게 고칩니다
if not RAW:
    print("에어코리아에서 받은 파일이 없습니다. 강의 자료의 파일로 1단계를 하십시오.")
else:
    parts = []
    for f in RAW:                                   # 파일 하나가 전국 약 48만 행이라 1분쯤 걸린다
        t = pd.read_excel(f, usecols=["지역", "망", "측정일시", "PM25"])
        parts.append(t[t["지역"].str.startswith("서울") & (t["망"] == "도시대기")])
    raw = pd.concat(parts)
    h = raw.groupby("측정일시")["PM25"].agg(PM25="mean", 측정소수="count").reset_index()
    date = pd.to_datetime(h["측정일시"].astype(str).str[:8])          # 2023010124 → 2023-01-01
    h["일시"] = date + pd.to_timedelta(h["측정일시"] % 100, unit="h")  # + 24시간 → 다음 날 0시
    h["PM25"] = h["PM25"].round(2)
    h = h[["측정일시", "일시", "PM25", "측정소수"]]
    h.to_csv("에어코리아_직접_줄인_파일.csv", index=False)   # 1단계의 AIR를 이 이름으로
    print(len(RAW), "개 파일 →", h.shape, h["일시"].min(), "~", h["일시"].max())

# ### MP-1
AIR = "에어코리아_서울_도시대기평균_시간별_2023_2026.csv"   # Colab 왼쪽 파일 창에 올린 두 파일
WEA = "asos_서울108_시간자료_2023_2026.csv"
air = pd.read_csv(AIR, parse_dates=["일시"])
wea = pd.read_csv(WEA, parse_dates=["일시"])
print("초미세먼지:", air.shape, air["일시"].min(), "~", air["일시"].max())
print("측정소 수:", air["측정소수"].min(), "~", air["측정소수"].max(), "곳")
print(air[["측정일시", "일시"]].iloc[22:24].to_string(index=False))  # 24시 = 다음 날 0시
print("날씨:", wea.shape, " 빈칸:", wea.isna().sum().to_dict())

wea.columns = ["지점", "지점명", "일시", "기온", "강수량", "풍속", "습도"]
wea["강수량"] = wea["강수량"].fillna(0)              # 비가 오지 않은 시각은 빈칸이다
air = air.merge(wea[["일시", "기온", "강수량", "풍속", "습도"]], on="일시")  # 같은 시각끼리
air["나쁨"] = (air["PM25"] > 35).astype(int)         # 예보 등급의 "나쁨"은 36㎍/㎥부터
print("합친 표:", air.shape, " 나쁨 비율:", round(air["나쁨"].mean(), 3))
print("매우 나쁨(76 이상) 비율:", round((air["PM25"] > 75).mean(), 3))
month = air.groupby(air["일시"].dt.month)["나쁨"].mean()
print(month.round(3).to_dict())

month.plot.bar(color="#E04B3C", figsize=(6, 3))
plt.xlabel("달")
plt.ylabel("나쁨 비율")
plt.xticks(rotation=0)
plt.show()

# ### MP-2
air["1시간전"] = air["PM25"].shift(1)              # 바로 앞 시각의 농도 (4단계에서 씀)
for c in ["기온", "습도", "풍속"]:                  # 그 시각까지 24시간의 평균
    air[c + "24"] = air[c].rolling(24).mean()
air["비24"] = (air["강수량"].rolling(24).sum() > 0).astype(int)  # 24시간 안에 비가 왔나
m = air["일시"].dt.month
air["겨울"] = m.isin([12, 1, 2]).astype(int)
air["봄"] = m.isin([3, 4, 5]).astype(int)           # 여름과 가을이 기준
X_COLS = ["기온24", "습도24", "풍속24", "비24", "겨울", "봄"]
air = air.dropna(subset=X_COLS)                     # 처음 23시각, 바람 기록이 빠진 때

yr = air["일시"].dt.year
past = air[yr == 2023]                              # 2023년으로 배우고
valid = air[yr == 2024]                             # 2024년으로 문턱값을 고르고
future = air[yr >= 2025]                            # 2025년 1월~2026년 4월로 한 번 시험
print("훈련", len(past), "/ 검증", len(valid), "/ 시험", len(future), "시각")
yt = future["나쁨"]
print("시험 기간 나쁨 비율:", round(yt.mean(), 3), " 기준선(늘 괜찮음):", round(1 - yt.mean(), 3))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
air_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
air_m.fit(past[X_COLS], past["나쁨"])
pt = air_m.predict_proba(future[X_COLS])[:, 1]
print("로지스틱 회귀(문턱값 0.5) 정확도:", round(accuracy_score(yt, pt >= 0.5), 3),
      " 재현율:", round(recall_score(yt, pt >= 0.5), 3))

# ### MP-3
pv = air_m.predict_proba(valid[X_COLS])[:, 1]       # 검증 기간(2024년)의 확률
yv = valid["나쁨"]
rows = []
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    tn, fp, fn, tp = confusion_matrix(yv, (pv >= th).astype(int)).ravel()
    rows.append([th, fp, fn, 3 * fn + fp])
air_cost = pd.DataFrame(rows, columns=["문턱값", "헛경보", "놓침", "손해(놓침×3+헛경보)"])
print(air_cost.to_string(index=False))
print("알림을 한 번도 보내지 않으면 손해:", 3 * yv.sum())
best = air_cost.loc[air_cost["손해(놓침×3+헛경보)"].idxmin(), "문턱값"]
print("검증으로 고른 문턱값:", best)

air_cost.plot(x="문턱값", y=["헛경보", "놓침"], marker="o", figsize=(6, 3),
              color=["#3F7CC0", "#E04B3C"])
plt.ylabel("시각 수 (2024년)")
plt.show()

# ### MP-4
plans = {"알림 안 보냄": np.zeros(len(yt), int), "늘 알림": np.ones(len(yt), int),
         "문턱값 0.5": (pt >= 0.5).astype(int), f"문턱값 {best}": (pt >= best).astype(int)}
for name, alert in plans.items():                   # 시험 기간에는 딱 한 번 판정한다
    tn, fp, fn, tp = confusion_matrix(yt, alert, labels=[0, 1]).ravel()
    print(f"{name:7s} 정확도 {(tp + tn) / len(yt):.3f}  재현율 {tp / (tp + fn):.3f}  "
          f"놓침 {fn:4d}  헛경보 {fp:5d}  손해 {3 * fn + fp}")

# ### MP-5
cols = X_COLS + ["1시간전"]
lag = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lag.fit(past[cols], past["나쁨"])
print("1시간 전 농도를 넣으면 정확도:", round(lag.score(future[cols], yt), 3))
same = (future["1시간전"] > 35).astype(int)         # "1시간 전과 같다"고 찍기
print("'1시간 전과 같다'고 찍으면:", round(accuracy_score(yt, same), 3))

# ### EX-6
th = 0.5
for name, pr in [("숫자 개수 하나", one.predict_proba(test[["숫자"]])[:, 1]),
                 ("로지스틱 회귀(단어)", prob)]:
    tn, fp, fn, tp = confusion_matrix(y, (pr >= th).astype(int)).ravel()
    print(f"{name}: 헛경보 {fp}  놓침 {fn}  손해 {10 * fp + fn}")
