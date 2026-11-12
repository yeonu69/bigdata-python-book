# 09장. 스팸일까, 아닐까: 분류 (v9판) — 실습 예제·코랩 실습·미니 프로젝트 전체 코드
# '# ### …' 조각 하나가 책의 [실습 예제 9-k] 또는 노트북에만 있는 [코랩 실습 9-k] 하나입니다. 끝에는 미니 프로젝트와 연습문제 9-6 정답 코드가 있습니다.
# 실행: cd ch09/data && python ../practice_ch09.py  (필요 라이브러리: pandas, numpy, scikit-learn, matplotlib)
# 미니 프로젝트는 data/ 폴더의 두 파일(에어코리아 서울 도시대기 평균, 기상청 서울 ASOS)을 씁니다.

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


# ### [실습 예제 9-1] [예제 9-1]을 파이썬 코드로 구현하기
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

TP, FP, FN, TN = 233, 81, 109, 468      # [예제 9-1]의 네 칸
y_true = np.array([1] * TP + [0] * FP + [1] * FN + [0] * TN)   # 실제(산 사람 1)
y_pred = np.array([1] * TP + [1] * FP + [0] * FN + [0] * TN)  # 판정(산다고 함 1)
print("승객:", len(y_true), " 산 사람:", y_true.sum(), " 산다고 함:", y_pred.sum())
print(confusion_matrix(y_true, y_pred))     # [[TN FP] [FN TP]] 순서
hand = {"정확도": 701 / 891, "정밀도": 233 / 314, "재현율": 233 / 342}
hand["F1"] = 2 * hand["정밀도"] * hand["재현율"] / (hand["정밀도"] + hand["재현율"])
func = {"정확도": accuracy_score(y_true, y_pred), "정밀도": precision_score(y_true, y_pred),
        "재현율": recall_score(y_true, y_pred), "F1": f1_score(y_true, y_pred)}
for k in hand:
    print(f"{k}: 손 계산 {hand[k]:.3f}  함수 {func[k]:.3f}")

# ### [실습 예제 9-2] [예제 9-2]를 파이썬 코드로 구현하기
def bayes(a, b, c):
    """a 사전확률, b 우도, c 헛증거 → 사후확률"""
    return a * b / (a * b + c * (1 - a))

print("free 하나:", round(bayes(0.126, 0.227, 0.012), 3))
print("검사 양성 한 번:", round(bayes(0.01, 0.9, 0.09), 3))
words = {"free": (0.227, 0.012), "call": (0.435, 0.039)}  # (스팸, 정상)에 든 비율
p = 0.126                                   # 처음 사전확률
for w, (b, c) in words.items():
    p = bayes(p, b, c)                      # 사후확률 → 다음 사전확률
    print(f'"{w}" 단어를 본 뒤: {p:.3f}')
import matplotlib.pyplot as plt
p, post = 0.01, [0.01]                     # 2.3의 검사: 유병률 1%
for k in range(5):                         # 다섯 번 연달아 양성
    p = bayes(p, 0.9, 0.09)
    post.append(p)
print([round(x, 3) for x in post])
plt.figure(figsize=(6, 3.2))
plt.plot(range(6), post, "o-", color="#0E6CA5")
for k, x in enumerate(post): plt.text(k, x + 0.04, f"{x:.2f}", ha="center", fontsize=9)
plt.xlabel("연달아 양성이 나온 횟수"); plt.ylabel("병일 확률(사후확률)"); plt.ylim(0, 1.12)
plt.title("양성이 거듭될수록 믿음이 커진다"); plt.grid(alpha=0.3)
plt.show()

# ### [실습 예제 9-3 ★] 문자 데이터 불러오기와 기준선
import numpy as np                             # 수 계산
import pandas as pd                            # 표 다루기
import matplotlib.pyplot as plt                # 그래프
pd.set_option("display.width", 120)            # 표를 한 줄에 넓게 찍는다
URL = ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
       "master/data/sms.tsv")
sms = pd.read_csv(URL, sep="\t", header=None, names=["label", "msg"], quoting=3)
print("모양(줄, 열):", sms.shape)
print("똑같은 문자:", sms.duplicated().sum(), "통")
for label, msg in sms.head(6).values:          # 처음 6줄의 (label, msg)
    print(f"{label:4s} | {msg[:70]}")          # 문자는 앞 70자만
sms = sms.drop_duplicates().reset_index(drop=True)    # 똑같은 문자는 하나만
print(sms["label"].value_counts().to_string())
print("모두 정상이라고 찍을 때 정확도:", round((sms["label"] == "ham").mean(), 3))
sms["스팸"] = (sms["label"] == "spam").astype(int)    # 스팸이면 1, 정상이면 0
sms["숫자"] = sms["msg"].str.count(r"\d")             # 문자 속 숫자(0~9)의 개수
sms["길이"] = sms["msg"].str.len()                    # 글자 수
print(sms.groupby("label")[["숫자", "길이"]].mean().round(1))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
sms["label"].value_counts().plot.bar(ax=ax[0], color=["#8EC5E8", "#E04B3C"])
ax[0].set_title("정상과 스팸의 수"); ax[0].tick_params(axis="x", rotation=0)
for lab, col in [("ham", "#8EC5E8"), ("spam", "#E04B3C")]:
    sms.loc[sms["label"] == lab, "숫자"].clip(upper=40).plot.hist(
        ax=ax[1], bins=20, alpha=0.7, color=col, label=lab)
ax[1].set_title("문자 속 숫자 개수"); ax[1].legend()
plt.show()

# ### [코랩 실습 9-1 ☆] 숫자 개수로 나누어 보기
bins = [-1, 0, 2, 5, 10, 20, 100]              # 구간의 경계
group = pd.cut(sms["숫자"], bins)               # 문자마다 어느 구간인지
table = sms.groupby(group, observed=True)["스팸"].agg(["size", "mean"])
print(table.round(3).to_string())
table["mean"].plot.bar(color="#E04B3C", figsize=(6, 3))
plt.xlabel("문자 속 숫자 개수"); plt.ylabel("스팸 비율")
plt.xticks(rotation=0); plt.show()
len_group = pd.qcut(sms["길이"], 5)            # 문자 수가 같은 다섯 구간
print(sms.groupby(len_group, observed=True)["스팸"].agg(["size", "mean"]).round(3).to_string())

# ### [실습 예제 9-4] 로지스틱 회귀를 문자 여덟 통으로 따라 하기
import numpy as np, pandas as pd
toy = pd.DataFrame({"숫자": [0, 1, 2, 3, 6, 8, 10, 12],
                    "스팸": [0, 0, 0, 1, 0, 1, 1, 1]})   # 예외 두 통 포함
print(toy.T)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

tb0, tb1 = -4.0, 0.8                    # 일단 정해 본 계수
toy["z"] = tb0 + tb1 * toy["숫자"]       # 회귀식
toy["확률"] = sigmoid(toy["z"])          # 0~1 확률
toy["판정"] = (toy["확률"] >= 0.5).astype(int)
print(toy.round(3))
print("맞힌 수:", (toy["판정"] == toy["스팸"]).sum(), "/ 8")
def log_loss_hand(b0, b1):
    p = sigmoid(b0 + b1 * toy["숫자"])
    y = toy["스팸"]
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

for b0, b1 in [(-4.0, 0.8), (-1.0, 0.1), (0.0, 0.0)]:
    print(f"b0={b0:5.1f}, b1={b1:.1f} → 로그 손실 {log_loss_hand(b0, b1):.3f}")
b0, b1, lr = 0.0, 0.0, 0.05             # lr: 한 걸음의 크기
x, y = toy["숫자"].values, toy["스팸"].values
for step in range(1, 3001):
    p = sigmoid(b0 + b1 * x)
    b0 -= lr * np.mean(p - y)           # 기울기 반대로 한 걸음
    b1 -= lr * np.mean((p - y) * x)
    if step in (1, 10, 100, 1000, 3000):
        loss = log_loss_hand(b0, b1)
        print(f"{step:5d}걸음: b0={b0:6.2f}, b1={b1:5.2f}, 손실 {loss:.3f}")
from sklearn.linear_model import LogisticRegression
tm = LogisticRegression(C=1e6)          # C를 크게: 규제 거의 없이
tm.fit(toy[["숫자"]], toy["스팸"])
print(f"사이킷런: b0={tm.intercept_[0]:.2f}, b1={tm.coef_[0][0]:.2f}")
import matplotlib.pyplot as plt
xs = np.linspace(0, 12, 100)
plt.figure(figsize=(6, 3.2))
plt.scatter(toy["숫자"], toy["스팸"], s=60, c=toy["스팸"],
            cmap="coolwarm", zorder=3, label="문자(1 = 스팸)")
plt.plot(xs, sigmoid(b0 + b1 * xs), color="#0E6CA5", label="손으로 찾은 곡선")
plt.axhline(0.5, ls="--", color="gray", lw=1)
plt.xlabel("숫자 개수"); plt.ylabel("스팸일 확률")
plt.legend(loc="center right"); plt.grid(alpha=0.3)
plt.show()

# ### [실습 예제 9-5 ★] 숫자 개수 하나로 로지스틱 회귀
from sklearn.model_selection import train_test_split
train, test = train_test_split(sms, test_size=0.25, random_state=42,
                               stratify=sms["스팸"])          # 스팸 비율을 지키며 나눈다
print("훈련", len(train), "통 / 시험", len(test), "통")
print("스팸 비율: 훈련", round(train["스팸"].mean(), 3), "/ 시험", round(test["스팸"].mean(), 3))
from sklearn.linear_model import LogisticRegression
one = LogisticRegression().fit(train[["숫자"]], train["스팸"])
b0, b1 = one.intercept_[0], one.coef_[0][0]
print(f"절편 {b0:.2f}, 기울기 {b1:.3f}, 오즈비 {np.exp(b1):.2f}")
print("확률이 0.5가 되는 숫자 개수:", round(-b0 / b1, 1))
for k in [0, 2, 5, 10, 15]:
    p = 1 / (1 + np.exp(-(b0 + b1 * k)))       # 시그모이드
    print(f"숫자 {k:2d}개 → 스팸일 확률 {p:.3f}")
three = test[["msg", "숫자"]].head(3)
three["모델 확률"] = one.predict_proba(three[["숫자"]])[:, 1].round(3)
three["판정"] = one.predict(three[["숫자"]])
three["msg"] = three["msg"].str[:40]
print(three.to_string())
two = LogisticRegression(max_iter=1000).fit(train[["숫자", "길이"]], train["스팸"])
for name, m, cols in [("숫자 하나", one, ["숫자"]), ("숫자+길이", two, ["숫자", "길이"])]:
    print(f"{name}: 시험 정확도 {m.score(test[cols], test['스팸']):.3f}")
print("숫자+길이의 계수:", two.coef_[0].round(3), " 오즈비:", np.exp(two.coef_[0]).round(3))
x = np.arange(0, 21)                           # 숫자 0개부터 20개까지
p = 1 / (1 + np.exp(-(b0 + b1 * x)))           # 한꺼번에 계산
rate = train.groupby(train["숫자"].clip(upper=20))["스팸"].mean()   # 실제 스팸 비율
plt.figure(figsize=(6, 3))
plt.scatter(rate.index, rate.values, color="gray", s=15, label="실제 비율(훈련)")
plt.plot(x, p, color="#E04B3C", label="모델의 확률")
plt.axhline(0.5, color="gray", linestyle="--")  # 문턱값 0.5
plt.xlabel("문자 속 숫자 개수"); plt.ylabel("스팸일 확률"); plt.legend(); plt.show()

# ### [실습 예제 9-6 ★] 혼동행렬과 정밀도, 재현율
from sklearn.metrics import confusion_matrix
pred = one.predict(test[["숫자"]])             # 문턱값 0.5로 판정한 0, 1
y = test["스팸"]                               # 정답
print(confusion_matrix(y, pred))               # [[TN FP] [FN TP]]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("정확도", round(accuracy_score(y, pred), 3), " 정밀도", round(precision_score(y, pred), 3))
print("재현율", round(recall_score(y, pred), 3), " F1", round(f1_score(y, pred), 3))
missed = test[(y == 1) & (pred == 0)]          # 실제 스팸인데 정상이라고 한 문자
print("놓친 스팸", len(missed), "통, 숫자 개수 평균", round(missed["숫자"].mean(), 1))
for msg in missed["msg"].head(4):
    print("-", msg[:75])
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["정상", "스팸"],
                                        cmap="Blues", colorbar=False)
plt.xlabel("모델의 판정"); plt.ylabel("실제"); plt.show()

# ### [실습 예제 9-7] 나이브 베이즈를 문자 네 통으로 따라 하기
from sklearn.feature_extraction.text import CountVectorizer
train4 = ["free prize call", "free call now",
          "call me later", "see you later"]
lab4 = np.array([1, 1, 0, 0])           # 1 = 스팸
cv4 = CountVectorizer().fit(train4)
X4 = cv4.transform(train4).toarray()
cnt = pd.DataFrame({"스팸": X4[lab4 == 1].sum(axis=0),     # 스팸 줄만 더하기
                    "정상": X4[lab4 == 0].sum(axis=0)},
                   index=cv4.get_feature_names_out())
print(cnt.T)
V = len(cnt)                            # 단어 종류 수
prob4 = (cnt + 1) / (cnt.sum() + V)     # (식 9-7), α = 1
print("단어 종류", V, "/ 범주별 단어 수", cnt.sum().to_dict())
print(prob4.round(3).T)
new = ["free", "call", "later"]         # 새 문자
prior = {"스팸": 0.5, "정상": 0.5}        # 사전확률
score = {c: prior[c] * np.prod([prob4.loc[w, c] for w in new]) for c in prior}
total = sum(score.values())
for c in score:
    terms = " × ".join(f"{prob4.loc[w, c]:.3f}" for w in new)
    print(f"{c}: {prior[c]} × {terms} = {score[c]:.5f} → 확률 {score[c] / total:.3f}")
from sklearn.naive_bayes import MultinomialNB
nb4 = MultinomialNB(alpha=1.0).fit(X4, lab4)
pp = nb4.predict_proba(cv4.transform(["free call later"]))
print("사이킷런 [정상, 스팸] 확률:", pp.round(3))

# ### [실습 예제 9-8 ★] 단어로 판정하기: 나이브 베이즈와 로지스틱 회귀
from sklearn.feature_extraction.text import CountVectorizer
toy = ["Free entry, call now", "Call me later", "free free prize"]
v = CountVectorizer().fit(toy)
print("사전:", list(v.get_feature_names_out()))
print(v.transform(toy).toarray())
vec = CountVectorizer()                        # 단어마다 몇 번 나왔는지 센다
A_tr = vec.fit_transform(train["msg"])         # 훈련 문자로 단어 사전을 만들고 센다
A_te = vec.transform(test["msg"])              # 시험 문자는 같은 사전으로 센다
print("훈련 문자표:", A_tr.shape, "/ 시험 문자표:", A_te.shape)
print("0이 아닌 칸의 비율:", round(A_tr.nnz / (A_tr.shape[0] * A_tr.shape[1]), 4))
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(A_tr, train["스팸"])                  # 나이브 베이즈
lr = LogisticRegression(max_iter=2000).fit(A_tr, train["스팸"])  # 로지스틱 회귀
print("나이브 베이즈가 센 단어 수(정상, 스팸):", nb.feature_count_.sum(axis=1).astype(int))
rows = []
for name, m, X in [("숫자 개수 하나", one, test[["숫자"]]),
                   ("나이브 베이즈", nb, A_te), ("로지스틱 회귀", lr, A_te)]:
    pr = m.predict(X)
    rows.append([name, accuracy_score(y, pr), precision_score(y, pr),
                 recall_score(y, pr), f1_score(y, pr)])
result = pd.DataFrame(rows, columns=["모델", "정확도", "정밀도", "재현율", "F1"])
print(result.round(3).to_string(index=False))
result.set_index("모델")[["정확도", "재현율"]].plot.barh(figsize=(6, 3),
                                                    color=["#8EC5E8", "#E04B3C"])
plt.axvline(0.874, color="gray", linestyle="--")   # 기준선: 모두 정상이라고 찍기
plt.xlim(0, 1); plt.show()

# ### [실습 예제 9-9 ★] 문턱값 바꾸기
prob = lr.predict_proba(A_te)[:, 1]            # 둘째 열 = 스팸일 확률
print(prob[:5].round(3))
print("확률 0.5 이상인 문자:", (prob >= 0.5).sum(), "통 / predict()의 스팸 판정:", lr.predict(A_te).sum(), "통")
rows = []
for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
    tn, fp, fn, tp = confusion_matrix(y, (prob >= th).astype(int)).ravel()
    rows.append([th, fp, fn, 10 * fp + fn])
cost = pd.DataFrame(rows, columns=["문턱값", "헛경보", "놓침", "손해"])
print(cost.to_string(index=False))
best = cost.loc[cost["손해"].idxmin()]         # 손해가 가장 작은 줄
print("고른 문턱값:", best["문턱값"], " 손해:", int(best["손해"]))
for w in [1, 10]:                              # 헛경보의 무게를 바꾸면?
    c = w * cost["헛경보"] + cost["놓침"]
    print(f"헛경보 × {w:2d}: 고른 문턱값 {cost.loc[c.idxmin(), '문턱값']}")
ths = np.linspace(0.05, 0.95, 19)              # 0.05부터 0.95까지 19개
fp_n = [((prob >= t) & (y == 0)).sum() for t in ths]
fn_n = [((prob < t) & (y == 1)).sum() for t in ths]
plt.figure(figsize=(6, 3))
plt.plot(ths, fp_n, marker="o", color="#3F7CC0", label="헛경보")
plt.plot(ths, fn_n, marker="o", color="#E04B3C", label="놓침")
plt.xlabel("문턱값"); plt.ylabel("문자 수"); plt.legend(); plt.show()

# ### [실습 예제 9-10 ★] 모델이 배운 단어 들여다보기
words = vec.get_feature_names_out()            # 열 순서대로 단어 이름
order = np.argsort(lr.coef_[0])                # 계수가 작은 것부터의 순서
print("정상 쪽:", list(words[order[:8]]))
print("스팸 쪽:", list(words[order[-8:]][::-1]))
low = sms["msg"].str.lower()                   # 대소문자를 같게
for w in ["lor", "da", "&lt;#&gt;"]:
    has = low.str.contains(w, regex=False) if w.startswith("&") else low.str.contains(rf"\b{w}\b")
    print(f"'{w}' 든 문자: 스팸 {has[sms['스팸'] == 1].sum()}통, 정상 {has[sms['스팸'] == 0].sum()}통")
i = int(np.argmax(prob))                       # 스팸 확률이 가장 높은 시험 문자
row = A_te[i].toarray()[0]                     # 그 문자의 단어 횟수
used = np.nonzero(row)[0]                      # 나온 단어의 열 번호
part = pd.Series(lr.coef_[0][used] * row[used], index=words[used]).sort_values(ascending=False)
print(test["msg"].iloc[i][:80])
print(part.head(6).round(2).to_string())
print("절편", round(lr.intercept_[0], 2), "+ 단어 점수 합", round(part.sum(), 2), "= z")
idx = list(order[:5]) + list(order[-5:])       # 정상 쪽 5개 + 스팸 쪽 5개
top = pd.Series(lr.coef_[0][idx], index=words[idx])
top.plot.barh(color=["#3F7CC0"] * 5 + ["#E04B3C"] * 5, figsize=(5, 4))
plt.xlabel("계수 (오른쪽일수록 스팸 쪽)"); plt.show()

# ### [코랩 실습 9-2 ☆] 데이터가 적을 때는 누가 이기나
small = train.sample(200, random_state=0)      # 훈련 문자 가운데 200통만
v2 = CountVectorizer().fit(small["msg"])
for name, m in [("나이브 베이즈", MultinomialNB()),
                ("로지스틱 회귀", LogisticRegression(max_iter=2000))]:
    m.fit(v2.transform(small["msg"]), small["스팸"])
    print(name, "F1", round(f1_score(y, m.predict(v2.transform(test["msg"]))), 3))
print("200통 속 스팸:", small["스팸"].sum(), "통")
def f1_small(n, seed):
    s = train.sample(n, random_state=seed)
    v = CountVectorizer().fit(s["msg"])
    out = []
    for m in [MultinomialNB(), LogisticRegression(max_iter=2000)]:
        m.fit(v.transform(s["msg"]), s["스팸"])
        out.append(f1_score(y, m.predict(v.transform(test["msg"]))))
    return out

rows = []
for n in [50, 200, 1000]:
    scores = np.array([f1_small(n, seed) for seed in range(10)])   # 10번 × 2모델
    rows.append([n, *scores.mean(axis=0).round(3), *scores.std(axis=0).round(3)])
small_res = pd.DataFrame(rows, columns=["훈련 문자 수", "나이브 베이즈 F1", "로지스틱 F1",
                                        "NB 표준편차", "LR 표준편차"])
print(small_res.to_string(index=False))
plt.figure(figsize=(6, 3))
for col, c in [("나이브 베이즈 F1", "#E04B3C"), ("로지스틱 F1", "#3F7CC0")]:
    plt.plot(small_res["훈련 문자 수"], small_res[col], marker="o", color=c, label=col)
plt.xscale("log"); plt.xlabel("훈련 문자 수(로그 눈금)"); plt.ylabel("평균 F1")
plt.legend(); plt.show()

# ### [코랩 실습 9-3 ☆] ROC 곡선과 AUC
from sklearn.metrics import roc_curve, roc_auc_score
probs = {"숫자 개수 하나": one.predict_proba(test[["숫자"]])[:, 1],
         "로지스틱 회귀(단어)": prob,
         "나이브 베이즈(단어)": nb.predict_proba(A_te)[:, 1]}
for name, pr in probs.items():
    print(f"{name:12s} AUC {roc_auc_score(y, pr):.3f}")
fpr, tpr, th = roc_curve(y, prob)              # 문턱값마다 (헛경보 비율, 재현율)
k = np.argmin(np.abs(th - 0.5))                # 문턱값 0.5에 가장 가까운 점
print(f"문턱값 {th[k]:.2f}: 헛경보 비율 {fpr[k]:.3f}, 재현율 {tpr[k]:.3f}")
print("곡선의 점 수:", len(th))
plt.figure(figsize=(5, 4))
for (name, pr), c in zip(probs.items(), ["#888888", "#3F7CC0", "#E04B3C"]):
    f, t, _ = roc_curve(y, pr)
    plt.plot(f, t, color=c, label=f"{name} ({roc_auc_score(y, pr):.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")   # 찍기 수준
plt.xlabel("헛경보 비율"); plt.ylabel("재현율"); plt.legend(fontsize=9); plt.show()

# ### [실습 예제 9-11 ★] 틀린 문자 읽기와 새 문자 판정
pred_nb = nb.predict(A_te)
p_nb = nb.predict_proba(A_te)[:, 1]
for kind, mask in [("헛경보", (y == 0) & (pred_nb == 1)), ("놓침", (y == 1) & (pred_nb == 0))]:
    print(f"[{kind}] {mask.sum()}통")
    for msg, p in zip(test["msg"][mask].head(3), p_nb[mask.values][:3]):
        print(f"  {p:.2f} | {msg[:65]}")
from sklearn.pipeline import make_pipeline
spam_filter = make_pipeline(CountVectorizer(), MultinomialNB())
spam_filter.fit(train["msg"], train["스팸"])    # 글자를 그대로 넣는다
print("시험 F1:", round(f1_score(y, spam_filter.predict(test["msg"])), 3))
new = ["Congratulations! You have won a free prize. Call 09061701461 now",
       "Are we still meeting for lunch tomorrow?",
       "URGENT: your account is locked, reply with your PIN",
       "Free parking near the station today",
       "Sorry, I'll call you later"]
for msg, p in zip(new, spam_filter.predict_proba(new)[:, 1]):
    print(f"{p:.3f} {'스팸' if p >= 0.5 else '정상'} | {msg}")

# ### [미니 프로젝트 9] 날씨만 보고 서울 초미세먼지 "나쁨"을 맞힐 수 있을까
AIR = "에어코리아_서울_도시대기평균_시간별_2023_2026.csv"   # 코랩 파일 창에 올린 두 파일
WEA = "asos_서울108_시간자료_2023_2026.csv"
air = pd.read_csv(AIR, parse_dates=["일시"])
wea = pd.read_csv(WEA, parse_dates=["일시"])
print("초미세먼지:", air.shape, air["일시"].min(), "~", air["일시"].max())
print(air[["측정일시", "일시"]].iloc[22:24].to_string(index=False))  # 24시 = 다음 날 0시
print("날씨:", wea.shape, " 빈칸:", wea.isna().sum().to_dict())
wea.columns = ["지점", "지점명", "일시", "기온", "강수량", "풍속", "습도"]
wea["강수량"] = wea["강수량"].fillna(0)              # 비가 오지 않은 시각은 빈칸이다
air = air.merge(wea[["일시", "기온", "강수량", "풍속", "습도"]], on="일시")  # 같은 시각끼리
air["나쁨"] = (air["PM25"] > 35).astype(int)         # 예보 등급의 "나쁨"은 36㎍/㎥부터
print("합친 표:", air.shape, " 나쁨 비율:", round(air["나쁨"].mean(), 3))
month = air.groupby(air["일시"].dt.month)["나쁨"].mean()
print(month.round(3).to_dict())
month.plot.bar(color="#E04B3C", figsize=(6, 3))
plt.xlabel("달"); plt.ylabel("나쁨 비율"); plt.xticks(rotation=0); plt.show()
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
past, valid, future = air[yr == 2023], air[yr == 2024], air[yr >= 2025]
print("훈련", len(past), "/ 검증", len(valid), "/ 시험", len(future), "시각")
yt = future["나쁨"]
print("시험 기간 나쁨 비율:", round(yt.mean(), 3), " 기준선(늘 괜찮음):", round(1 - yt.mean(), 3))
from sklearn.preprocessing import StandardScaler
air_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
air_m.fit(past[X_COLS], past["나쁨"])
pt = air_m.predict_proba(future[X_COLS])[:, 1]
print("로지스틱 회귀(문턱값 0.5) 정확도:", round(accuracy_score(yt, pt >= 0.5), 3),
      " 재현율:", round(recall_score(yt, pt >= 0.5), 3))
print(pd.Series(air_m[-1].coef_[0], index=X_COLS).round(2).to_dict())
pv = air_m.predict_proba(valid[X_COLS])[:, 1]       # 검증 기간(2024년)의 확률
yv = valid["나쁨"]
rows = []
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    tn, fp, fn, tp = confusion_matrix(yv, (pv >= th).astype(int)).ravel()
    rows.append([th, fp, fn, 3 * fn + fp])
air_cost = pd.DataFrame(rows, columns=["문턱값", "헛경보", "놓침", "손해"])
print(air_cost.to_string(index=False))
print("알림을 한 번도 보내지 않으면 손해:", 3 * yv.sum())
best = air_cost.loc[air_cost["손해"].idxmin(), "문턱값"]
print("검증으로 고른 문턱값:", best)
air_cost.plot(x="문턱값", y=["헛경보", "놓침"], marker="o", figsize=(6, 3),
              color=["#3F7CC0", "#E04B3C"])
plt.ylabel("시각 수 (2024년)"); plt.show()
plans = {"알림 안 보냄": np.zeros(len(yt), int), "늘 알림": np.ones(len(yt), int),
         "문턱값 0.5": (pt >= 0.5).astype(int), f"문턱값 {best}": (pt >= best).astype(int)}
for name, alert in plans.items():                   # 시험 기간에는 딱 한 번 판정한다
    tn, fp, fn, tp = confusion_matrix(yt, alert, labels=[0, 1]).ravel()
    print(f"{name:7s} 정확도 {(tp + tn) / len(yt):.3f}  재현율 {tp / (tp + fn):.3f}  "
          f"놓침 {fn:4d}  헛경보 {fp:5d}  손해 {3 * fn + fp}")
cols = X_COLS + ["1시간전"]
lag = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lag.fit(past[cols], past["나쁨"])
print("1시간 전 농도를 넣으면 정확도:", round(lag.score(future[cols], yt), 3))
same = (future["1시간전"] > 35).astype(int)         # "1시간 전과 같다"고 찍기
print("'1시간 전과 같다'고 찍으면:", round(accuracy_score(yt, same), 3))

# ### [연습문제 9-6] 정확도가 높은 모델이 더 좋은가
th = 0.5
for name, pr in [("숫자 개수 하나", one.predict_proba(test[["숫자"]])[:, 1]),
                 ("로지스틱 회귀(단어)", prob)]:
    tn, fp, fn, tp = confusion_matrix(y, (pr >= th).astype(int)).ravel()
    print(f"{name}: 헛경보 {fp}  놓침 {fn}  손해 {10 * fp + fn}")
