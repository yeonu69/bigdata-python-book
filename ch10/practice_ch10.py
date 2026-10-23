# 10장. 스무고개와 집단지성: 결정나무와 앙상블 (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 10-k' 조각 하나가 책의 [파이썬 실습 10-k] 하나입니다. MP-k는 [미니 프로젝트 10]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: pandas, numpy, scikit-learn, matplotlib)

# ### 10-1
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

URL = ("https://raw.githubusercontent.com/khatrideepti/"
       "Bank-Marketing-Data-Analysis/master/bank-full.csv")
bank = pd.read_csv(URL, sep=";")
y = (bank["y"] == "yes").astype(int)                  # 가입 1, 안 함 0
print(bank.shape, " 가입:", y.sum(), "명, 비율", round(y.mean(), 3))
print("빈칸:", bank.isna().sum().sum(), " 'unknown' 연락 수단:",
      (bank["contact"] == "unknown").sum())

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
m = bank["month"].map({k: i for i, k in enumerate(MONTHS)}).values
bank["year"] = 2008 + np.r_[0, (m[1:] < m[:-1]).cumsum()]  # 달이 거꾸로 가면 새해
by_year = bank.groupby("year").agg(
    기록=("y", "size"), 가입률=("y", lambda s: (s == "yes").mean()),
    연락수단_모름=("contact", lambda s: (s == "unknown").mean()))
print(by_year.round(3))

cut = int(len(bank) * 0.75)                            # 앞 75%로 배우고 뒤 25%로 시험
for name, row in [("훈련 끝", bank.iloc[cut - 1]), ("시험 시작", bank.iloc[cut])]:
    print(name, row["year"], row["month"])

by_year["가입률"].plot.bar(color="#E04B3C", figsize=(5, 3))
plt.xlabel("연도")
plt.ylabel("가입률")
plt.xticks(rotation=0)
plt.show()

# ### 10-2
def gini(p):                                           # 가입 비율 p인 무리의 지니 불순도
    return 1 - p**2 - (1 - p)**2

root = gini(y.mean())
print("나누기 전 지니:", round(root, 4))
questions = {"지난 캠페인 성공?": bank["poutcome"] == "success",
             "나이 60세 초과?": bank["age"] > 60,
             "주택대출 있음?": bank["housing"] == "yes",
             "3월에 연락?": bank["month"] == "mar"}
for q, mask in questions.items():
    a, b = y[mask], y[~mask]                           # 예 / 아니요 두 무리
    after = (len(a) * gini(a.mean()) + len(b) * gini(b.mean())) / len(y)
    print(f"{q:10s} 예 {len(a):5d}명 {a.mean():.3f} / 아니요 {len(b):5d}명 {b.mean():.3f}"
          f"  지니 감소 {root - after:.4f}")

succ = questions["지난 캠페인 성공?"]                   # 둘째 질문까지 내려가 보기
for name, grp, q2, label in [("성공한 고객", succ, bank["housing"] == "yes", "주택대출"),
                             ("나머지 고객", ~succ, bank["age"] > 60, "60세 초과")]:
    for ans in [True, False]:
        g = y[grp & (q2 == ans)]
        print(f"{name} → {label} {'예' if ans else '아니요'}: {len(g):5d}명, 가입률 {g.mean():.3f}")

# ### 10-3
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score

X = pd.get_dummies(bank.drop(columns=["y", "year", "duration"]),
                   dtype=int)                          # 통화 시간은 뺀다(3.7)
Xtr, Xte = X.iloc[:cut], X.iloc[cut:]                  # 시간 순서로 나누기
ytr, yte = y.iloc[:cut], y.iloc[cut:]
print("특징", X.shape[1], "개 / 훈련", len(Xtr), "/ 시험", len(Xte))
print("가입률: 훈련", round(ytr.mean(), 3), " 시험", round(yte.mean(), 3),
      " 시험 기간 가입:", yte.sum(), "건")
print("모두 '안 함'으로 찍은 정확도:", round(1 - yte.mean(), 3))

def auc(model, Xs, ys):                                # 확률로 줄 세우는 능력
    return roc_auc_score(ys, model.predict_proba(Xs)[:, 1])

t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
print(export_text(t3, feature_names=list(X.columns), max_depth=1))
print("깊이 3 나무: 정확도", round(t3.score(Xte, yte), 3),
      " AUC", round(auc(t3, Xte, yte), 3))
for mo in ["mar", "oct"]:                              # 훈련 기간의 3월과 10월
    k = (bank["month"].iloc[:cut] == mo).values
    print(mo, k.sum(), "건, 연도", sorted(set(bank["year"].iloc[:cut][k])),
          " 가입률", round(ytr[k].mean(), 3))

from sklearn.tree import plot_tree
plt.figure(figsize=(10, 4))
plot_tree(t3, max_depth=2, feature_names=list(X.columns), class_names=["안 함", "가입"],
          filled=True, impurity=False, proportion=True, fontsize=9)
plt.show()

# ### 10-4
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print("끝까지 자란 나무: 깊이", full.get_depth(), " 잎", full.get_n_leaves())
print("훈련 정확도", round(full.score(Xtr, ytr), 3),
      "/ 시험 정확도", round(full.score(Xte, yte), 3),
      "/ 시험 AUC", round(auc(full, Xte, yte), 3))

rows = []
for depth in [1, 2, 3, 4, 6, 8, 10, 15, 20]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    rows.append([depth, auc(t, Xtr, ytr), auc(t, Xte, yte)])
curve = pd.DataFrame(rows, columns=["깊이", "훈련 AUC", "시험 AUC"])
print(curve.round(3).to_string(index=False))

curve.plot(x="깊이", y=["훈련 AUC", "시험 AUC"], marker="o",
           color=["#3F7CC0", "#E04B3C"], figsize=(6, 3))
plt.ylabel("AUC")
plt.show()

# ### 10-5
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(5, shuffle=True, random_state=0)  # 훈련 기간을 다섯 조각으로
for n in [10, 50, 100, 200, 400]:
    s = cross_val_score(DecisionTreeClassifier(min_samples_leaf=n, random_state=0),
                        Xtr, ytr, cv=cv, scoring="roc_auc")
    print(f"잎마다 최소 {n:3d}명: 검증 AUC {s.mean():.3f}")

# ### 10-6
from sklearn.ensemble import RandomForestClassifier

def forest(**kw):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  random_state=0, n_jobs=-1, **kw).fit(Xtr, ytr)

kinds = {"똑같은 나무": forest(max_features=None, bootstrap=False),
         "배깅": forest(max_features=None),
         "랜덤 포레스트": forest(max_features="sqrt")}
rows = []
for name, f in kinds.items():
    one = [roc_auc_score(yte, t.predict_proba(Xte.values)[:, 1])
           for t in f.estimators_[:30]]                # 숲 속 나무 30그루를 하나씩
    first = {X.columns[t.tree_.feature[0]] for t in f.estimators_}
    rows.append([name, np.mean(one), auc(f, Xte, yte), len(first)])
crowd = pd.DataFrame(rows, columns=["숲", "나무 하나(평균)", "숲 전체", "첫 질문 종류"])
print(crowd.round(3).to_string(index=False))

rf = kinds["랜덤 포레스트"]
probs = np.array([t.predict_proba(Xte.values)[:, 1] for t in rf.estimators_])
for k in [1, 3, 10, 30, 100, 300]:                     # 앞의 k그루만 평균하면
    print(f"나무 {k:3d}그루 평균: 시험 AUC {roc_auc_score(yte, probs[:k].mean(0)):.3f}")

crowd.set_index("숲")[["나무 하나(평균)", "숲 전체"]].plot.barh(
    color=["#8EC5E8", "#E04B3C"], figsize=(6, 3))
plt.xlim(0.5, 0.75)
plt.xlabel("시험 AUC")
plt.legend(loc="lower right")
plt.show()

# ### 10-7
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def boost(rate, Xa=Xtr, ya=ytr):
    return GradientBoostingClassifier(n_estimators=300, learning_rate=rate,
                                      max_depth=3, subsample=0.8,
                                      random_state=0).fit(Xa, ya)

gb, fast = boost(0.05), boost(1.0)
slow_auc = [roc_auc_score(yte, p[:, 1]) for p in gb.staged_predict_proba(Xte)]
fast_auc = [roc_auc_score(yte, p[:, 1]) for p in fast.staged_predict_proba(Xte)]
for k in [1, 10, 50, 100, 300]:
    print(f"나무 {k:3d}그루: 학습률 0.05 → {slow_auc[k - 1]:.3f},"
          f" 학습률 1.0 → {fast_auc[k - 1]:.3f}")
print("학습률 0.05가 가장 좋았던 곳:", int(np.argmax(slow_auc)) + 1, "그루",
      round(max(slow_auc), 3))

pruned = DecisionTreeClassifier(min_samples_leaf=100, random_state=0)
pruned.fit(Xtr, ytr)                                   # 가지친 나무: 잎마다 고객 100명 이상
print("가지친 나무: 잎", pruned.get_n_leaves())
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
lr.fit(Xtr, ytr)
board = {"깊이 3 나무": t3, "가지친 나무": pruned, "로지스틱 회귀(9장)": lr,
         "랜덤 포레스트": rf, "그레이디언트 부스팅": gb}
rows = []
for name, model in board.items():
    p = model.predict_proba(Xte)[:, 1]
    top = np.argsort(-p)                               # 확률 높은 순서
    found = yte.values[top].cumsum() / yte.sum()
    rows.append([name, auc(model, Xte, yte), found[len(p) // 10 - 1],
                 found[len(p) * 3 // 10 - 1]])
score = pd.DataFrame(rows, columns=["모델", "시험 AUC", "상위 10%가 만난 가입자",
                                    "상위 30%가 만난 가입자"])
print(score.round(3).to_string(index=False))

plt.figure(figsize=(6, 3))
plt.plot(range(1, 301), slow_auc, color="#E04B3C", label="학습률 0.05")
plt.plot(range(1, 301), fast_auc, color="#3F7CC0", label="학습률 1.0")
plt.xlabel("더한 나무의 수")
plt.ylabel("시험 AUC")
plt.legend()
plt.show()

# ### 10-8
from sklearn.inspection import permutation_importance

Xd = pd.get_dummies(bank.drop(columns=["y", "year"]), dtype=int)  # 통화 시간 포함
gbd = boost(0.05, Xd.iloc[:cut], ytr)
print("통화 시간을 넣으면: 시험 AUC", round(auc(gbd, Xd.iloc[cut:], yte), 3),
      " 통화 시간의 중요도", round(gbd.feature_importances_[Xd.columns.get_loc("duration")], 3))
mins = pd.cut(bank["duration"] / 60, [0, 1, 3, 5, 10, 100], right=False)
print(y.groupby(mins, observed=True).mean().round(3).to_string())

imp = pd.Series(gb.feature_importances_, X.columns)    # 통화 시간 없는 부스팅
print("불순도 중요도 상위 5\n", imp.sort_values(ascending=False).head(5).round(3))
pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5,
                            random_state=0, n_jobs=-1)
perm = pd.Series(pi.importances_mean, X.columns)
print("섞었을 때 떨어지는 AUC 상위 5\n", perm.sort_values(ascending=False).head(5).round(4))
mar = bank[bank["month"] == "mar"]
print("3월 기록:", len(mar), "건, 연도", sorted(mar["year"].unique().tolist()),
      " 가입률", round((mar["y"] == "yes").mean(), 3))

perm.sort_values().tail(8).plot.barh(color="#E04B3C", figsize=(5, 3.5))
plt.xlabel("섞었을 때 떨어진 시험 AUC")
plt.show()

# ### 10-9
from sklearn.model_selection import train_test_split
Rtr, Rte, rtr, rte = train_test_split(X, y, test_size=0.25, stratify=y,
                                      random_state=42)   # 세 해를 섞어 무작위로
mixed = DecisionTreeClassifier(min_samples_leaf=100, random_state=0).fit(Rtr, rtr)
print("무작위로 나누면: 가지친 나무", round(auc(mixed, Rte, rte), 3))
lrr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Rtr, rtr)
print("무작위로 나누면: 부스팅", round(auc(boost(0.05, Rtr, rtr), Rte, rte), 3),
      " 로지스틱", round(auc(lrr, Rte, rte), 3))
c8 = int(len(X) * 0.8)                                 # 자르는 곳을 80%로 옮기면
lr8 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
lr8.fit(X.iloc[:c8], y.iloc[:c8])
print("80%에서 자르면: 부스팅", round(auc(boost(0.05, X.iloc[:c8], y.iloc[:c8]),
                                         X.iloc[c8:], y.iloc[c8:]), 3),
      " 로지스틱", round(auc(lr8, X.iloc[c8:], y.iloc[c8:]), 3))
print("80% 지점:", bank["year"].iloc[c8], bank["month"].iloc[c8])

# ### Q-1
def gini(p):
    return 1 - p**2 - (1 - p)**2
before = gini(4 / 10)                                  # 10명 가운데 4명 가입
after_A = 4 / 10 * gini(3 / 4) + 6 / 10 * gini(1 / 6)  # 질문 A로 나눈 뒤
after_B = 5 / 10 * gini(2 / 5) + 5 / 10 * gini(2 / 5)  # 질문 B로 나눈 뒤
print(round(before, 3), round(after_A, 3), round(after_B, 3))
print("질문 A의 값:", round(before - after_A, 3), " 질문 B의 값:", round(before - after_B, 3))

# ### Q-2
from math import comb
def vote(n, p):                                        # n명이 저마다 p로 맞힐 때 다수결이 맞을 확률
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))
for n in [1, 3, 11, 101, 501]:
    print(f"{n:3d}명: p=0.6 → {vote(n, 0.6):.3f},  p=0.55 → {vote(n, 0.55):.3f}")

# ### Q-3
for rho in [0, 0.3, 1]:                                # 나무끼리의 상관
    print(f"상관 {rho}:", [round(rho + (1 - rho) / B, 3) for B in [1, 10, 300]])

# ### Q-4
import numpy as np
y4 = np.array([1, 0, 0, 0])                            # 고객 넷 가운데 첫째만 가입
F = np.full(4, y4.mean())                              # 처음에는 모두 0.25
for m in range(3):
    r = y4 - F                                         # 잔차 = 실제 - 예측
    F = F + 0.5 * r                                    # 잔차를 맞힌 나무를 학습률 0.5만큼 더한다
    print(f"나무 {m + 1}그루 뒤 예측:", F.round(3), " 첫째 고객의 잔차:", round(y4[0] - F[0], 3))

# ### MP-1
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital",
        "occupation", "relationship", "race", "sex", "capital_gain",
        "capital_loss", "hours", "country", "income"]
ADULT = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
adult = pd.read_csv(ADULT, header=None, names=COLS)
print(adult.shape, " 똑같은 행:", adult.duplicated().sum())
print("'?'로 적힌 칸:", (adult == "?").sum()[lambda s: s > 0].to_dict())
ya = (adult["income"] == ">50K").astype(int)
print("5만 달러 초과 비율:", round(ya.mean(), 3))
print(adult.groupby("sex")["income"].apply(lambda s: (s == ">50K").mean()).round(3).to_string())

# ### MP-2
XA = pd.get_dummies(adult.drop(columns=["income"]), dtype=int)
Atr, Ate, atr, ate = train_test_split(XA, ya, test_size=0.25, stratify=ya,
                                      random_state=42)
print("특징", XA.shape[1], "개, 기준선 정확도", round(1 - ate.mean(), 3))
a3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Atr, atr)
print(export_text(a3, feature_names=list(XA.columns), max_depth=1))
arf = RandomForestClassifier(n_estimators=200, random_state=0,
                             n_jobs=-1).fit(Atr, atr)        # 기본 설정 그대로
for name, mdl in [("깊이 3 나무", a3), ("랜덤 포레스트", arf)]:
    print(f"{name}: 정확도 {mdl.score(Ate, ate):.3f}  AUC {auc(mdl, Ate, ate):.3f}")

# ### MP-3
imp_a = pd.Series(arf.feature_importances_, XA.columns)
print("불순도 중요도 상위 4\n", imp_a.sort_values(ascending=False).head(4).round(3))
base = auc(arf, Ate, ate)
rng = np.random.default_rng(0)
for col in ["fnlwgt", "age", "capital_gain", "hours"]:     # 한 열만 섞어 보기
    S = Ate.copy()
    S[col] = rng.permutation(S[col].values)
    print(f"{col:13s} 섞으면 AUC {base:.3f} → {auc(arf, S, ate):.3f}")

# ### MP-4
pa = arf.predict_proba(Ate)[:, 1] >= 0.5
for s in ["Male", "Female"]:
    k = (adult.loc[Ate.index, "sex"] == s).values
    hit = pa[k] & (ate.values[k] == 1)
    print(f"{s:6s}: 실제 5만 달러 초과 {ate.values[k].sum():4d}명, 모델이 찾은 비율 "
          f"{hit.sum() / ate.values[k].sum():.3f}")

# ### EX-6
drop = [c for c in X.columns if c == "day" or c.startswith("month_")]
gb2 = boost(0.05, Xtr.drop(columns=drop), ytr)
print("뺀 열:", len(drop), "개 / 시험 AUC:", round(auc(gb2, Xte.drop(columns=drop), yte), 3))
