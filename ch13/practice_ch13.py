# 13장. 몸을 가진 AI (개정판) — 05절 파이썬 실습과 종합 프로젝트 전체 코드
# '# ### 13-k' 조각 하나가 책의 [파이썬 실습 13-k] 하나입니다.
# MP-k는 [미니 프로젝트 13] 종합 프로젝트의 출발 코드, EX-6은 연습문제 정답 코드입니다.
# 실행: python3 practice_rev.py  (필요 라이브러리: numpy, pandas, scikit-learn, matplotlib, scipy, holidays)

# ### 13-1
import os, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib            # 코랩: 한글 글꼴을 자동으로 설정
except ImportError:                        # 나눔고딕 파일이 이미 있는 컴퓨터
    import glob
    import matplotlib.font_manager as fm
    for f in glob.glob(os.path.expanduser("~/.fonts/Nanum*.ttf")):
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

HAR = "UCI HAR Dataset"
MIRROR = "https://raw.githubusercontent.com/bidhanbaray/UCI_HAR_Dataset/master/"
CH = [f"{s}_{a}" for s in ["body_acc", "body_gyro", "total_acc"] for a in "xyz"]

def get_har():                             # 필요한 파일만 깃허브 사본에서 받는다
    files = []
    for s in ["train", "test"]:
        files += [f"{s}/y_{s}.txt", f"{s}/subject_{s}.txt"]
        files += [f"{s}/Inertial Signals/{c}_{s}.txt" for c in CH]
    for f in files:
        if not os.path.exists(f"{HAR}/{f}"):
            os.makedirs(os.path.dirname(f"{HAR}/{f}"), exist_ok=True)
            urllib.request.urlretrieve(MIRROR + f.replace(" ", "%20"), f"{HAR}/{f}")

def load(split):                           # (윈도우 수, 시간 128칸, 채널 9)
    sig = np.stack([np.loadtxt(f"{HAR}/{split}/Inertial Signals/{c}_{split}.txt")
                    for c in CH], axis=2)
    y = np.loadtxt(f"{HAR}/{split}/y_{split}.txt").astype(int) - 1
    s = np.loadtxt(f"{HAR}/{split}/subject_{split}.txt").astype(int)
    return sig, y, s

get_har()
sig_tr, y_tr, s_tr = load("train")
sig_te, y_te, s_te = load("test")
print("훈련:", sig_tr.shape, " 시험:", sig_te.shape)
print("한 윈도우 =", sig_tr.shape[1] / 50, "초 (50Hz)")

ACT = ["걷기", "계단 오르기", "계단 내려가기", "앉기", "서기", "눕기"]
y_all, s_all = np.r_[y_tr, y_te], np.r_[s_tr, s_te]
print("훈련에 나온 사람:", sorted(map(int, set(s_tr))))
print("시험에 나온 사람:", sorted(map(int, set(s_te))))
per = pd.Series(s_all).value_counts().sort_index()
print("사람 수:", len(per), " 한 사람의 윈도우 수:", per.min(), "~", per.max())
print(pd.Series(y_all).map(dict(enumerate(ACT))).value_counts().reindex(ACT).to_string())

color = ["#C8234B" if s in set(s_te) else "#9a9a9a" for s in per.index]
per.plot.bar(color=color, figsize=(7.5, 3))
plt.xlabel("사람 번호 (빨강: 시험용 9명, 회색: 훈련용 21명)")
plt.ylabel("윈도우 수")
plt.xticks(rotation=0, fontsize=10)
plt.show()

# ### 13-2
sig_all = np.r_[sig_tr, sig_te]
grav = pd.DataFrame([sig_all[y_all == i][:, :, 6:9].mean(axis=(0, 1)) for i in range(6)],
                    index=ACT, columns=["x", "y", "z"])
body = np.linalg.norm(sig_all[:, :, 0:3], axis=2).std(axis=1)    # 몸 흔들림의 표준편차
grav["흔들림(중앙값)"] = [np.median(body[y_all == i]) for i in range(6)]
print(grav.round(3))

fr = np.fft.rfftfreq(128, 1 / 50)          # 0, 0.39, 0.78, ... 25Hz
def spec(x):                               # 윈도우마다 주파수별 세기
    x = x - x.mean(axis=1, keepdims=True)
    return np.abs(np.fft.rfft(x * np.hanning(128), axis=1))

mag = np.linalg.norm(sig_all[:, :, 6:9], axis=2)   # 전체 가속도의 크기
dom = fr[1:][spec(mag)[:, 1:].argmax(axis=1)]      # 가장 센 주파수
walk = np.median(dom[y_all == 0])
print("주파수 간격:", round(fr[1], 3), "Hz,  가장 높은 주파수:", fr[-1], "Hz")
print("걷기의 가장 센 주파수(중앙값):", round(walk, 2), "Hz → 1분에", round(walk * 60), "걸음")

grav[["x", "y", "z"]].plot.bar(color=["#C8234B", "#9a9a9a", "#4d4d4d"], figsize=(6.5, 3.2))
plt.xticks(range(6), ["걷기", "오르기", "내려가기", "앉기", "서기", "눕기"], rotation=0)
plt.ylabel("평균 가속도 (g)")
plt.legend(title="축", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.show()

for i, c in [(0, "#C8234B"), (3, "#7a7a7a")]:   # 걷기와 앉기의 평균 스펙트럼
    plt.plot(fr, spec(mag)[y_all == i].mean(axis=0), color=c, label=ACT[i])
plt.xlim(0, 10)
plt.xlabel("주파수 (Hz, 1초에 몇 번 흔들리나)")
plt.ylabel("세기 (평균)")
plt.legend()
plt.gcf().set_size_inches(6, 3)
plt.show()

# ### 13-3
from sklearn.tree import DecisionTreeClassifier

def feats(sig):                            # 신호 11개 × 통계 10개 = 특징 110개
    chans = [sig[:, :, j] for j in range(9)] + [np.linalg.norm(sig[:, :, 6:9], axis=2),
                                                np.linalg.norm(sig[:, :, 3:6], axis=2)]
    cols = []
    for c in chans:
        Sp = spec(c)
        band = lambda lo, hi: (Sp[:, (fr >= lo) & (fr < hi)] ** 2).sum(axis=1)
        q75, q25 = np.percentile(c, [75, 25], axis=1)
        cols += [c.mean(1), c.std(1), c.min(1), c.max(1), q75 - q25,
                 np.sqrt((c ** 2).mean(1)), fr[1:][Sp[:, 1:].argmax(1)],
                 band(0.3, 3), band(3, 10), band(10, 25.1)]
    return np.column_stack(cols)

F_tr, F_te = feats(sig_tr), feats(sig_te)
print("특징 표:", F_tr.shape)
j = 9 * 10 + 1                             # 10번째 신호(전체 가속도 크기)의 표준편차
stump = DecisionTreeClassifier(max_depth=1).fit(F_tr[:, [j]], y_tr <= 2)
print("질문 하나: 흔들림 >", round(stump.tree_.threshold[0], 4), "이면 움직이는 중")
print("움직임/멈춤 정확도:", stump.score(F_te[:, [j]], y_te <= 2))
acc_d = {}
for d in [2, 3, 4, 6, 8]:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(F_tr, y_tr)
    acc_d[d] = t.score(F_te, y_te)
    print(f"깊이 {d} 나무: 6동작 정확도 {acc_d[d]:.3f}, 마디 {t.tree_.node_count}개")

pd.Series(acc_d).plot(marker="o", color="#C8234B", figsize=(5, 3))
plt.xlabel("나무의 깊이 (질문을 몇 번 이어 하나)")
plt.ylabel("6동작 정확도")
plt.ylim(0.5, 1)
plt.show()

# ### 13-4
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
p_rf = rf.fit(F_tr, y_tr).predict(F_te)
print("처음 보는 9명으로 시험:", round((p_rf == y_te).mean() * 100, 2), "%")

F_all = np.r_[F_tr, F_te]                  # 사람을 섞어서 무작위로 나누면
A, B, ya, yb = train_test_split(F_all, y_all, test_size=len(y_te) / len(y_all),
                                random_state=42, stratify=y_all)
rf2 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(A, ya)
print("윈도우를 무작위로 나눠 시험:", round(rf2.score(B, yb) * 100, 2), "%")

acc = pd.Series({s: (p_rf[s_te == s] == y_te[s_te == s]).mean() * 100 for s in sorted(set(s_te))})
print("사람별 정확도:", acc.round(1).to_dict())
cm = confusion_matrix(y_te, p_rf)
print("앉기→서기로 착각:", cm[3, 4], "  서기→앉기로 착각:", cm[4, 3])

acc.sort_values().plot.bar(color="#9a9a9a", figsize=(6, 3), rot=0)
plt.axhline((p_rf == y_te).mean() * 100, color="#C8234B", linestyle="--")
plt.xlabel("시험용 사람 번호 (점선: 9명 평균)")
plt.ylabel("정확도 (%)")
plt.ylim(60, 100)
plt.show()

# ### 13-5
rng = np.random.default_rng(0)
people = np.array(sorted(set(s_tr)))
RF = lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
curve = {}
for k in [1, 3, 6, 12, 21]:                # 훈련에 쓰는 사람 수를 늘리면
    accs = []
    for rep in range(5 if k < 21 else 1):
        m = np.isin(s_tr, rng.choice(people, k, replace=False))
        accs.append((RF().fit(F_tr[m], y_tr[m]).predict(F_te) == y_te).mean() * 100)
    curve[k] = np.mean(accs)
    print(f"{k:2d}명: 평균 {np.mean(accs):.1f}% ({min(accs):.1f}~{max(accs):.1f}%)")
for k in [3, 21]:                          # 윈도우 800개를 3명에게서, 21명에게서
    accs = []
    for rep in range(5):
        idx = np.where(np.isin(s_tr, rng.choice(people, k, replace=False)))[0]
        idx = rng.choice(idx, 800, replace=False)
        accs.append((RF().fit(F_tr[idx], y_tr[idx]).predict(F_te) == y_te).mean() * 100)
    print(f"윈도우 800개, {k}명에게서: {np.mean(accs):.1f}%")

pd.Series(curve).plot(marker="o", color="#C8234B", figsize=(5, 3))
plt.xlabel("훈련에 쓴 사람 수")
plt.ylabel("처음 보는 9명 정확도 (%)")
plt.show()

# ### 13-6
df = pd.read_csv("https://raw.githubusercontent.com/lingchenlijing/AI4I-2020-"
                 "Predictive-Maintenance-Dataset/master/data/ai4i2020.csv")
df.columns = [c.replace("﻿", "") for c in df.columns]
print(df.shape, " 고장:", df["Machine failure"].sum(), "건")
df["dT"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]
print(df[["Air temperature [K]", "dT", "Torque [Nm]"]].agg(["mean", "std"]).round(2))
rules = {"HDF": (df.dT < 8.6) & (df["Rotational speed [rpm]"] < 1380),
         "PWF": (df.power < 3500) | (df.power > 9000),
         "OSF": df.strain > df.Type.map({"L": 11000, "M": 12000, "H": 13000})}
for m, rule in rules.items():
    print(f"{m}: 규칙 {rule.sum()}건, 라벨 {df[m].sum()}건, 겹침 {(rule & (df[m] == 1)).sum()}건")

from sklearn.metrics import f1_score, precision_score, recall_score
num = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
       "Torque [Nm]", "Tool wear [min]"]
df["TypeN"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
base, phys = num + ["TypeN"], num + ["TypeN", "dT", "power", "strain"]
y = df["Machine failure"].values
i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
print("시험 2,000건 가운데 고장:", y[i_te].sum(), "건, 모두 정상으로 찍으면",
      round((y[i_te] == 0).mean() * 100, 1), "%")
for name, cols in [("센서값 그대로", base), ("물리 특징 추가", phys)]:
    m = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    m.fit(df.loc[i_tr, cols], y[i_tr])
    pr = m.predict_proba(df.loc[i_te, cols])[:, 1]
    p = (pr >= 0.5).astype(int)
    print(f"{name}: 정밀도 {precision_score(y[i_te], p):.3f} 재현율 {recall_score(y[i_te], p):.3f}"
          f" F1 {f1_score(y[i_te], p):.3f}")

pb = pd.cut(df.power, [0, 3500, 5000, 7000, 9000, 20000])
rate = df.groupby(pb, observed=True)["Machine failure"].agg(["size", "mean"])
print(rate.round(3).to_string())
(rate["mean"] * 100).plot.bar(color=["#C8234B", "#9a9a9a", "#9a9a9a", "#9a9a9a", "#C8234B"],
                              figsize=(6, 3), rot=0)
plt.xticks(range(5), ["3,500 미만", "3,500~5,000", "5,000~7,000", "7,000~9,000", "9,000 초과"],
           fontsize=10)
plt.xlabel("기계적 출력 (W)")
plt.ylabel("고장률 (%)")
plt.show()

# ### 13-7
def angle(sig):                            # 중력과 휴대폰 각 축 사이의 각도
    g = sig[:, :, 6:9].mean(axis=1)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(g, -1, 1)))

ang = angle(sig_all)
print(pd.DataFrame([ang[y_all == i].mean(axis=0) for i in (3, 4, 5)],
                   index=ACT[3:], columns=["x축", "y축", "z축"]).round(1))
G_tr, G_te = np.c_[F_tr, angle(sig_tr)], np.c_[F_te, angle(sig_te)]
p_g = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(G_tr, y_tr).predict(G_te)
cm_g = confusion_matrix(y_te, p_g)
print("각도 3개를 더한 정확도:", round((p_g == y_te).mean() * 100, 2), "%")
print("앉기·서기 혼동:", cm[3, 4] + cm[4, 3], "→", cm_g[3, 4] + cm_g[4, 3], "건")

# ### 13-8
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

for s in ["train", "test"]:                # 연구팀이 만든 특징 561개 (약 92MB)
    f = f"{HAR}/{s}/X_{s}.txt"
    if not os.path.exists(f):
        urllib.request.urlretrieve(MIRROR + f"{s}/X_{s}.txt", f)
X561_tr, X561_te = np.loadtxt(f"{HAR}/train/X_train.txt"), np.loadtxt(f"{HAR}/test/X_test.txt")
lr561 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
print("특징 561개 + 로지스틱 회귀:", round(lr561.fit(X561_tr, y_tr).score(X561_te, y_te) * 100, 2), "%")
lr110 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
print("특징 110개 + 로지스틱 회귀:", round(lr110.fit(F_tr, y_tr).score(F_te, y_te) * 100, 2), "%")

# ### 13-9
cost = lambda yt, p: 10 * ((p == 0) & (yt == 1)).sum() + ((p == 1) & (yt == 0)).sum()
i_fit, i_val = train_test_split(i_tr, test_size=0.25, random_state=0, stratify=y[i_tr])
mv = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
pv = mv.fit(df.loc[i_fit, phys], y[i_fit]).predict_proba(df.loc[i_val, phys])[:, 1]
rows = []
for t in [0.1, 0.2, 0.3, 0.5, 0.7]:        # 문턱값은 훈련 안의 검증용 2,000건으로 고른다
    p = (pv >= t).astype(int)
    rows.append([t, p.sum(), (p & (y[i_val] == 1)).sum(), cost(y[i_val], p)])
val = pd.DataFrame(rows, columns=["문턱값", "경보", "맞힌 고장", "손해(놓침×10+헛경보)"])
print(val.to_string(index=False))
best = val.loc[val.iloc[:, 3].idxmin(), "문턱값"]
p = (pr >= best).astype(int)              # pr: 실습 13-6의 물리 특징 모델이 시험 2,000건에 낸 확률
print(f"고른 문턱값 {best} → 시험: 경보 {p.sum()}건, 맞힌 고장 {(p & (y[i_te] == 1)).sum()}건,"
      f" 재현율 {recall_score(y[i_te], p):.3f}, 손해 {cost(y[i_te], p)}")
p5 = (pr >= 0.5).astype(int)
print(f"문턱값 0.5 그대로 → 시험: 경보 {p5.sum()}건, 손해 {cost(y[i_te], p5)}")

# ### 13-10
from sklearn.ensemble import IsolationForest

Xn = StandardScaler().fit_transform(df[num])
iso = IsolationForest(n_estimators=300, contamination=0.0339, random_state=42)
flag = iso.fit_predict(Xn) == -1           # 라벨 없이 "이상하다"고 고른 것
fail = df["Machine failure"] == 1
print("이상으로 고른 것:", flag.sum(), "건, 그 가운데 실제 고장:", (flag & fail).sum(), "건")
print("정밀도:", round((flag & fail).sum() / flag.sum() * 100, 1), "%  (전체 고장 비율 3.39%)")
modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print({m: f"{(flag & (df[m] == 1)).sum()}/{df[m].sum()}" for m in modes})

# ### 13-11
import pickle
from sklearn.neural_network import MLPClassifier

mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                                                    early_stopping=True, random_state=0))
mlp.fit(F_tr, y_tr)
n_par = sum(w.size for w in mlp[-1].coefs_) + sum(b.size for b in mlp[-1].intercepts_)
t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(F_tr, y_tr)
for name, model, n in [("깊이 3 나무", t3, None), ("신경망 64-32", mlp, n_par), ("랜덤 포레스트 200그루", rf, None)]:
    kb = len(pickle.dumps(model)) / 1024
    extra = f", 매개변수 {n:,}개 × 4바이트 = {n * 4 / 1024:.1f}KB" if n else ""
    print(f"{name}: 정확도 {model.score(F_te, y_te) * 100:.2f}%, 저장 파일 {kb:,.1f}KB{extra}")

# ### M-1
import numpy as np
a = np.array([0, 0.6, 0.8])               # 가만히 놓인 휴대폰의 가속도 (g)
size = np.linalg.norm(a)                   # (식 13-1)
print("크기:", round(size, 3), "g")
print("각도(°):", np.degrees(np.arccos(a / size)).round(1))   # (식 13-2)

# ### M-2
fs, N = 50, 128
freq = np.fft.rfftfreq(N, 1 / fs)          # 칸마다의 주파수
print("가장 높은 주파수:", fs / 2, "Hz,  칸 간격:", fs / N, "Hz")
print("4~6번째 칸:", freq[4:7].round(3), "Hz")
print("5번째 칸 → 1분에", round(freq[5] * 60), "걸음")

# ### M-3
for torque, rpm in [(40, 1500), (60, 1450)]:
    P = torque * 2 * np.pi * rpm / 60      # (식 13-5)
    print(f"토크 {torque}Nm, {rpm}rpm → {P:,.0f}W")

# ### M-4
layers = [110, 64, 32, 6]                  # 입력 110개 → 64 → 32 → 6동작
n = sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))
print("매개변수:", f"{n:,}", "개")
print("4바이트 실수:", round(n * 4 / 1024, 1), "KB,  1바이트 정수:", round(n / 1024, 1), "KB")

# ### MP-0
# 종합 프로젝트 공통: 기상청 서울(108) 시간자료 → 하루 단위
w = pd.read_csv("asos_서울108_시간자료_2023_2026.csv",
                usecols=["일시", "기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)"])
w.columns = ["일시", "기온", "강수량", "풍속", "습도"]
w["일시"] = pd.to_datetime(w["일시"])
print("시간자료:", len(w), "행,", w["일시"].min(), "~", w["일시"].max())
print("빈칸:", w.isna().sum().to_dict())
w["강수량"] = w["강수량"].fillna(0)          # 비가 안 온 시각은 빈칸으로 적혀 있다
day = w.groupby(w["일시"].dt.normalize()).agg(기온=("기온", "mean"), 강수량=("강수량", "sum"),
                                             풍속=("풍속", "mean"), 습도=("습도", "mean"))
print("하루 단위:", day.shape)

# ### MP-RAW
# ☆ 포털에서 받은 역별 원자료를 서울 전체 하루 단위로 합치기 (강의 자료의 파일을 쓰면 건너뜁니다)
import glob
RAW = sorted(glob.glob("서울교통공사_역별*승하차인원*.csv"))     # 받은 파일 이름에 맞게 고칩니다
if not RAW:
    print("포털에서 받은 파일이 없습니다. 강의 자료의 서울 전체 파일로 하십시오.")
else:
    parts = []
    for f in RAW:
        t = pd.read_csv(f, encoding="cp949")                    # 포털 파일의 한글은 cp949
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
    print(len(RAW), "개 파일 →", city.shape, city.index[0][0], "~", city.index[-1][0])

# ### MP-A
# 주제 A: 전력 수요 (한국전력거래소, 전국, 시간별 MW)
pw = pd.read_csv("kpx_시간별_전국_전력수요량_2021_2025.csv", parse_dates=["날짜"])
hours = [f"{h}시" for h in range(1, 25)]
pw["GWh"] = pw[hours].mean(axis=1) * 24 / 1000        # 하루 전력량
A = pw.set_index("날짜")[["GWh"]].join(day, how="inner")
print(A.shape, A.index.min().date(), "~", A.index.max().date())
print(A.groupby(pd.cut(A["기온"], [-15, 0, 10, 20, 25, 35]), observed=True)["GWh"].mean().round(0))
A.plot.scatter(x="기온", y="GWh", s=6, color="#9a9a9a", figsize=(5, 3.2))
plt.xlabel("서울 하루 평균 기온 (°C)")
plt.ylabel("전국 하루 전력 수요 (GWh)")
plt.show()

# ### MP-B
# 주제 B: 지하철 (서울교통공사, 모든 역을 더한 하루·시간대별 승하차)
sub = pd.read_csv("서울교통공사_서울전체_일별_시간대별_승하차_2023_2024.csv", parse_dates=["날짜"])
B = sub[sub["구분"] == "승차"].set_index("날짜")[["13-14시", "합계"]] / 10000   # 만 명
month = B.groupby(B.index.to_period("M")).mean().round(1)
print(month.loc["2024-02":"2024-06"])
month.index = month.index.strftime("%y.%m")          # 23.01 꼴의 글자로
month["13-14시"].plot(marker="o", color="#C8234B", figsize=(6, 3), rot=0)
plt.ylabel("13~14시 승차 (만 명, 하루 평균)")
plt.xlabel("")
plt.show()

# ### MP-C
# 주제 C: 초미세먼지 (에어코리아 확정자료, 서울 도시대기 측정소 25곳 평균)
air = pd.read_csv("에어코리아_서울_도시대기평균_시간별_2023_2026.csv", parse_dates=["일시"])
air["날"] = (air["일시"] - pd.Timedelta(hours=1)).dt.normalize()   # 24시 기록은 그날 것
C = air.groupby("날")["PM25"].mean().to_frame("PM25").join(day, how="inner")
C["내일나쁨"] = (C["PM25"].shift(-1) > 35).astype(int)            # 36㎍/㎥ 이상이면 "나쁨"
C = C.iloc[:-1]
print(C.shape, " 다음 날이 '나쁨' 이상인 날:", round(C["내일나쁨"].mean() * 100, 1), "%")
print(C.groupby(C.index.month)["PM25"].mean().round(1).to_dict())

# ### EX-6
twf = df["TWF"] == 1
print("공구 마모 고장(TWF):", twf.sum(), "건, 마모 시간",
      df.loc[twf, "Tool wear [min]"].min(), "~", df.loc[twf, "Tool wear [min]"].max(), "분")
rule = df["Tool wear [min]"] >= 200
print("마모 200분 이상:", rule.sum(), "건, 그 가운데 TWF:", (rule & twf).sum(), "건")
print("정밀도:", round((rule & twf).sum() / rule.sum() * 100, 1), "%, 재현율:",
      round((rule & twf).sum() / twf.sum() * 100, 1), "%")
