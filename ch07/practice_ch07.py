# 07장. 끼리끼리 모으기: 군집화 (개정판 v3) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 7-k' 조각 하나가 책의 [파이썬 실습 7-k] 하나입니다. MP-k는 [미니 프로젝트 7]의 단계,
# EX-k는 연습문제 정답 코드, S6-k는 06절 숫자를 확인하는 덤 코드입니다(책에는 코드가 실리지 않음).
# 실행: python3 practice_v3.py  (필요 라이브러리: pandas, numpy, scikit-learn, scipy, matplotlib)

# ### 7-1
import os
import urllib.request
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

DATA = "Significant_Earthquakes.csv"
URL = ("https://raw.githubusercontent.com/Abdullah-Moataz/Earthquakes/"
       "cf0d033d35f5d45eef6175bab6c60829363d6de4/Significant_Earthquakes.csv")
if not os.path.exists(DATA):                       # 파일이 없을 때만 내려받기(약 21MB)
    urllib.request.urlretrieve(URL, DATA)

df = pd.read_csv(DATA, index_col=0)                # 첫 열은 행 번호
print("처음 읽은 행:", len(df), " 같은 id가 다시 나온 행:", df["id"].duplicated().sum())
df["time"] = pd.to_datetime(df["time"], format="ISO8601")
# 같은 지진이 여러 번 실렸으면 가장 나중에 고친(updated) 기록 하나만 남기고, 다시 시간 순서로
df = df.sort_values("updated").drop_duplicates("id", keep="last").sort_values("time")
print("중복을 지운 뒤:", len(df))
print(df["type"].value_counts().head(4).to_string())

eq = df[(df["type"] == "earthquake")
        & df["latitude"].between(-85, 85) & df["longitude"].between(-180, 180)]
print("지진만, 위치가 맞는 것:", len(eq))
eq = eq.dropna(subset=["depth", "mag"]).copy()      # 깊이·규모가 빈 행 제외
eq["region"] = eq["place"].fillna("unknown").str.split(", ").str[-1]
print("분석할 지진:", len(eq), "건,", eq["time"].min().date(), "~", eq["time"].max().date())
print("규모를 잰 방법:", eq["magType"].nunique(), "가지,",
      eq["magType"].value_counts().head(2).to_dict())

year = eq["time"].dt.year
print("1972년:", (year == 1972).sum(), "건  1973년:", (year == 1973).sum(), "건")
small = eq["mag"] < 5.5                            # 규모 5.5 미만 = 작은 지진
print("작은 지진의 비율(%)  1972년까지:", round(small[year < 1973].mean() * 100, 1),
      " 1973년부터:", round(small[year >= 1973].mean() * 100, 1))

decade = year // 10 * 10                           # 1973 → 1970
eq.groupby(decade)["mag"].size().plot.bar(color="#9a9a9a", figsize=(6, 3))
plt.xlabel("연대")
plt.ylabel("기록된 지진 수")
plt.xticks(rotation=45)
plt.show()

# ### 7-2
print(eq[["depth", "mag"]].describe().round(2))
depth_class = pd.cut(eq["depth"], [-10, 70, 300, 800],
                     labels=["얕은(70km 이하)", "중간(70~300km)", "깊은(300km 초과)"])
print((depth_class.value_counts(normalize=True) * 100).round(1).to_string())
d = eq["depth"].round(0)
print("깊이가 정확히 10km:", round((d == 10).mean() * 100, 1), "%,  33km:",
      round((d == 33).mean() * 100, 1), "%")
print("깊이와 규모의 상관계수:", round(eq["depth"].corr(eq["mag"]), 3))
late = eq[year >= 1973]                            # 기록이 고르게 된 뒤
print("1973년부터  규모 6 이상:", (late["mag"] >= 6).sum(), "건, 7 이상:", (late["mag"] >= 7).sum(), "건")
print(pd.cut(eq["depth"], range(200, 701, 50)).value_counts().sort_index().to_string())

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].hist(eq["depth"], bins=range(0, 720, 20), color="#9a9a9a")
ax[0].set_yscale("log")                            # 깊은 쪽의 작은 봉우리가 보이도록
ax[0].set_xlabel("깊이(km)")
ax[0].set_ylabel("지진 수(로그 눈금)")
ax[1].hist(eq["mag"], bins=np.arange(5, 9.6, 0.1), color="#E04B3C")
ax[1].set_xlabel("규모")
plt.tight_layout()
plt.show()

# ### 7-3
pts = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
C = pts[[0, 3]].copy()                             # 처음 중심: 점 1과 점 4
for step in range(1, 10):
    d2 = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # 점-중심 거리의 제곱
    a = d2.argmin(axis=1)                          # ① 배정: 가장 가까운 중심
    print(f"{step}회: 중심 {C.round(2).tolist()}, 배정 {a.tolist()}, "
          f"WCSS {d2.min(axis=1).sum():.2f}")
    newC = np.array([pts[a == j].mean(axis=0) for j in range(2)])  # ② 갱신: 평균
    if np.allclose(newC, C):                       # 중심이 그대로면 멈춤
        break
    C = newC

plt.figure(figsize=(4, 4))
plt.scatter(pts[:, 0], pts[:, 1], c=np.where(a == 1, "#E04B3C", "#9a9a9a"), s=80)
plt.scatter(C[:, 0], C[:, 1], marker="x", s=150, color="black")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ### 7-4
def dist(p, q):                                    # 유클리드 거리
    return np.sqrt(((p - q) ** 2).sum())

q = np.array([[30, 5.0], [40, 8.0], [80, 5.0]])    # [깊이(km), 규모] 지진 1, 2, 3
print("표준화 전  1-2:", round(dist(q[0], q[1]), 2), " 1-3:", round(dist(q[0], q[2]), 2))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = eq[["depth", "mag"]].to_numpy()
scaler = StandardScaler().fit(X)                   # 평균과 표준편차를 배운다
print("평균:", scaler.mean_.round(2), " 표준편차:", scaler.scale_.round(2))
z = scaler.transform(q)                            # (식 7-3)을 적용한다
print("표준화 후  1-2:", round(dist(z[0], z[1]), 2), " 1-3:", round(dist(z[0], z[2]), 2))
print("깊이 15km, 규모 6.84인 지진의 z:", scaler.transform([[15, 6.84]]).round(2))

raw = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)   # 표준화 없이 묶으면
for depth, mag in sorted(raw.cluster_centers_.tolist()):
    print(f"표준화 없이 묶은 중심: 깊이 {depth:6.1f}km, 규모 {mag:.2f}")

plt.figure(figsize=(6, 3))
plt.scatter(X[::20, 0], X[::20, 1], c=raw.labels_[::20], cmap="Greys", vmin=-1, s=6)
plt.xlabel("깊이(km)")
plt.ylabel("규모")
plt.show()

# ### 7-5
Xs = scaler.transform(X)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
centers = scaler.inverse_transform(km.cluster_centers_)   # 원래 단위로 되돌리기
print("표준화된 중심:", km.cluster_centers_.round(2).tolist())
print("원래 단위의 중심:", centers.round(2).tolist())

def name(c):                                       # 중심을 보고 사람이 붙이는 이름
    return "C 깊은" if c[0] > 200 else ("B 얕고 강한" if c[1] > 5.7 else "A 얕고 보통")

eq["군집"] = [name(centers[l]) for l in km.labels_]
summary = eq.groupby("군집").agg(지진수=("mag", "size"), 깊이평균=("depth", "mean"),
                                 깊이최소=("depth", "min"), 규모평균=("mag", "mean"),
                                 규모최소=("mag", "min"), 규모최대=("mag", "max"))
summary["비율%"] = (summary["지진수"] / len(eq) * 100).round(1)
print(summary.round(2).to_string())

period = np.where(year < 1973, "1972년까지", "1973년부터")   # 기록이 크게 는 해
print((eq.groupby(period)["군집"].value_counts(normalize=True) * 100).round(1).unstack())
fixed = d.isin([10, 33])                           # 관례로 정해 넣은 깊이
print("정해 넣은 깊이의 비율(%):", (fixed.groupby(eq["군집"]).mean() * 100).round(1).to_dict())

COLOR = {"A 얕고 보통": "#9a9a9a", "B 얕고 강한": "#E04B3C", "C 깊은": "#222222"}
s = eq.sample(3000, random_state=1)
plt.figure(figsize=(6, 3.5))
for g, part in s.groupby("군집"):
    plt.scatter(part["depth"], part["mag"], s=6, color=COLOR[g], label=g)
plt.xlabel("깊이(km)")
plt.ylabel("규모")
plt.legend(markerscale=3)
plt.show()

# ### 7-6
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(42)
samp = rng.choice(len(Xs), 10000, replace=False)    # 실루엣은 1만 건 표본으로
rows = []
for k in range(2, 9):
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    rows.append([k, m.inertia_, silhouette_score(Xs[samp], m.labels_[samp])])
    if k == 2:
        print("k=2의 두 군집:", sorted(np.bincount(m.labels_).tolist(), reverse=True))
score = pd.DataFrame(rows, columns=["k", "WCSS", "실루엣"])
print(score.round({"WCSS": 0, "실루엣": 3}).to_string(index=False))

# 무리가 없는 데이터와 견주기 (k = 3)
U = rng.uniform(Xs.min(axis=0), Xs.max(axis=0), size=Xs.shape)   # 같은 범위에 고르게 뿌린 점
S = np.column_stack([rng.permutation(Xs[:, 0]),                 # 깊이와 규모를 따로 섞어
                     rng.permutation(Xs[:, 1])])                # 둘의 짝을 끊은 지진
for label, data in [("실제 지진", Xs), ("고르게 뿌린 점", U), ("따로 섞은 지진", S)]:
    m = KMeans(n_clusters=3, n_init=10, random_state=42).fit(data)
    sizes = np.sort(np.bincount(m.labels_)) / len(data) * 100
    print(f"{label:8s} 실루엣 {silhouette_score(data[samp], m.labels_[samp]):.3f}"
          f"  군집 크기(%) {sizes.round(1).tolist()}")

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].plot(score["k"], score["WCSS"], "o-", color="#555555")
ax[0].set_xlabel("k")
ax[0].set_ylabel("WCSS")
ax[1].plot(score["k"], score["실루엣"], "o-", color="#E04B3C")
ax[1].set_xlabel("k")
ax[1].set_ylabel("실루엣 계수")
plt.tight_layout()
plt.show()

# ### 7-7
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(eq[["depth", "mag"]], eq["군집"])
print(export_text(tree, feature_names=["depth", "mag"], decimals=1))
print("군집을 맞힌 비율:", round(tree.score(eq[["depth", "mag"]], eq["군집"]) * 100, 2), "%")

# ### 7-8
from sklearn.metrics import adjusted_rand_score

eq["log깊이"] = np.log1p(eq["depth"].clip(lower=0))   # log(1 + 깊이)
XL = StandardScaler().fit_transform(eq[["log깊이", "mag"]])
kl = KMeans(n_clusters=3, n_init=10, random_state=42).fit(XL)
print(eq.groupby(kl.labels_).agg(지진수=("mag", "size"), 깊이중앙값=("depth", "median"),
                                 규모평균=("mag", "mean")).round(2).to_string())
print("원래 군집과 겹치는 정도(조정 랜드 지수):", round(adjusted_rand_score(km.labels_, kl.labels_), 2))

# ### 7-9
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

top12 = eq["region"].value_counts().head(12).index
sub = eq[eq["region"].isin(top12)]
prof = sub.groupby("region").agg(깊이평균=("depth", "mean"),
                                 중간깊이비율=("depth", lambda s: ((s > 70) & (s <= 300)).mean()),
                                 깊은비율=("depth", lambda s: (s > 300).mean()),
                                 규모평균=("mag", "mean")).loc[top12]
Z = linkage(StandardScaler().fit_transform(prof), method="ward")
prof["무리"] = fcluster(Z, t=3, criterion="maxclust")
print(prof.round(3).sort_values("무리").to_string())
plt.figure(figsize=(6, 4))
dendrogram(Z, labels=prof.index.tolist(), orientation="right", color_threshold=0,
           above_threshold_color="#555555")
plt.xlabel("합칠 때의 거리")
plt.show()

# ### 7-10
from sklearn.cluster import DBSCAN

big = eq[eq["mag"] >= 6].copy()                    # 규모 6 이상
rad = np.radians(big[["latitude", "longitude"]])
db = DBSCAN(eps=150 / 6371, min_samples=40,        # 반지름 150km 안에 40개 이상
            metric="haversine", algorithm="ball_tree").fit(rad)
big["zone"] = db.labels_                           # -1은 노이즈
print("규모 6 이상 지진:", len(big), "건")
print("찾은 밀집 지대:", big.loc[big.zone >= 0, "zone"].nunique(), "곳",
      " 노이즈 비율:", round((big.zone == -1).mean() * 100, 1), "%")
size = big[big.zone >= 0].groupby("zone").agg(n=("mag", "size"),
                                              지역=("region", lambda s: s.mode()[0]))
print(size.sort_values("n", ascending=False).head(5).to_string(index=False))
biggest = size["n"].idxmax()                       # 가장 큰 지대에는 어느 지역이?
print(big.loc[big.zone == biggest, "region"].value_counts().head(3).to_dict())

# ### S6-1
# 6.1 시작점에 따라 달라지는 답 (k = 5, 한 번씩만 돌리기)
runs = [KMeans(n_clusters=5, n_init=1, init="random", random_state=s).fit(Xs).inertia_
        for s in range(10)]
print("무작위 시작 10번의 WCSS:", sorted(np.round(runs).astype(int).tolist()))
print("서로 다른 값:", len(set(np.round(runs))), "가지")
pp = KMeans(n_clusters=5, n_init=1, init="k-means++", random_state=0).fit(Xs).inertia_
print("k-평균++ 한 번:", round(pp))

# ### S6-2
# 6.3 이상 탐지: 가장 가까운 중심까지의 거리가 가장 먼 0.5%
far = np.min(((Xs[:, None, :] - km.cluster_centers_[None]) ** 2).sum(axis=2), axis=1) ** 0.5
eq["중심거리"] = far
top = eq[far >= np.quantile(far, 0.995)].sort_values("중심거리", ascending=False)
print("가장 먼 0.5%:", len(top), "건")
print(top[["time", "mag", "depth", "region"]].head(6).to_string(index=False))

# ### S6-3
# 6.2 여진 묶기: 2011년 일본 주변 규모 5 이상 지진을 위치(100km)와 시간(30일)으로
win = eq[(year == 2011) & eq["latitude"].between(25, 50) & eq["longitude"].between(125, 155)].copy()
main = win.loc[win["mag"].idxmax()]
xk = (win["longitude"] - main["longitude"]) * 111.2 * np.cos(np.radians(38)) / 100
yk = (win["latitude"] - main["latitude"]) * 111.2 / 100
tk = (win["time"] - pd.Timestamp("2011-01-01", tz="UTC")).dt.total_seconds() / 86400 / 30
win["st"] = DBSCAN(eps=0.6, min_samples=10).fit(np.column_stack([xk, yk, tk])).labels_
seq = win[win["st"] == win.loc[main.name, "st"]]
after = seq["time"] - main["time"]
print("2011년 일본 주변:", len(win), "건, 본진", main["time"].date(), "규모", main["mag"])
print("본진과 한 묶음:", len(seq), "건, 첫날", ((after >= pd.Timedelta(0)) & (after < pd.Timedelta(days=1))).sum(),
      "건, 첫 주", ((after >= pd.Timedelta(0)) & (after < pd.Timedelta(days=7))).sum(),
      "건, 한 달 뒤부터", (after >= pd.Timedelta(days=30)).sum(), "건, 전진", (after < pd.Timedelta(0)).sum(), "건")

# ### MP-0
# [더 알아보기] 공공데이터포털에서 받은 원자료로 이 책의 CSV 파일 만들기
import holidays                                    # 공휴일 목록 (없으면 !pip install holidays)
RAW = "서울교통공사_역별 일별 시간대별 승하차인원 정보_20241231.csv"
raw = pd.read_csv(RAW, encoding="cp949")           # 한글이 cp949로 저장된 파일
print(raw.shape)
hours = raw.columns[6:].tolist()                   # '06시이전' ~ '24시이후'
raw["날짜"] = pd.to_datetime(raw["수송일자"])
kr = holidays.KR(years=2024)
off_day = (raw["날짜"].dt.dayofweek >= 5) | raw["날짜"].dt.date.map(lambda x: x in kr)
raw["날종류"] = np.where(off_day, "쉬는날", "평일")
raw["월"] = raw["날짜"].dt.month
# 환승역은 호선마다 따로 적혀 있으므로 역명으로 더한 뒤, 달·날종류마다 하루 평균
daily = raw.groupby(["역명", "날짜", "월", "날종류", "승하차구분"])[hours].sum().reset_index()
m = daily.groupby(["월", "날종류", "역명", "승하차구분"])[hours].mean().round(1).reset_index()
ndays = daily.groupby(["월", "날종류"])["날짜"].nunique().rename("날수").reset_index()
lines = (raw.groupby("역명")["호선"].unique()
         .map(lambda a: ",".join(sorted(a, key=lambda t: int(t[0])))).rename("호선"))
m = m.merge(ndays, on=["월", "날종류"]).merge(lines, left_on="역명", right_index=True)
m = m.rename(columns={"승하차구분": "구분"})
m.columns = [c.replace("시간대", "시") for c in m.columns]
m = m[["월", "날종류", "날수", "역명", "호선", "구분"] + [h.replace("시간대", "시") for h in hours]]
m.to_csv("서울지하철_역별_시간대별_월평균_승하차_2024.csv", index=False)
print(m.shape)

# ### MP-1
# 데이터 바꾸기: 다른 해의 파일을 쓸 때는 이 두 줄만 고칩니다.
PATH = "서울지하철_역별_시간대별_월평균_승하차_2024.csv"   # Colab 왼쪽 파일 창에 올린 파일
HOURS = slice("06시이전", "24시이후")                    # 시간대 열의 처음과 끝

sw = pd.read_csv(PATH)
print(sw.shape, "/ 빈칸", sw.isna().sum().sum(), "/ 역", sw["역명"].nunique(), "곳")
per_st = sw.groupby("역명").size()                    # 역마다 몇 줄인가 (12달 × 2 × 2 = 48)
print(per_st.value_counts().to_dict(), per_st[per_st != 48].to_dict())
H = sw.loc[:, HOURS].columns
month = sw[sw["날종류"] == "평일"].groupby("월")[H].sum() / 1000   # 달별 시간대 합(천 명)
print(month[["12-13시", "13-14시", "14-15시"]].round(0).astype(int).T.to_string())

month.T.plot(legend=False, color="#9a9a9a", figsize=(7, 3))
plt.plot(month.loc[4].values, color="#E04B3C", linewidth=2)   # 4월만 빨갛게
plt.xticks(range(len(H)), H, rotation=90)
plt.ylabel("평일 하루 평균(천 명)")
plt.show()

# ### MP-2
ok = sw[(sw["날종류"] == "평일") & (sw["월"] != 4)]          # 4월은 뺀다
w = ok.assign(**{h: ok[h] * ok["날수"] for h in H})           # 날수만큼 가중
prof = (w.groupby(["역명", "구분"])[H].sum()
        .div(w.groupby(["역명", "구분"])["날수"].sum(), axis=0)
        .unstack("구분"))                                     # 한 역 = 한 줄
prof.columns = [f"{b}_{h}" for h, b in prof.columns]
prof = prof[[f"{b}_{h}" for b in ["승차", "하차"] for h in H]]
total = prof.sum(axis=1)
print("승하차가 0인 역:", total[total == 0].index.tolist())
prof, total = prof[total > 0], total[total > 0]
share = prof.div(total, axis=0) * 100                         # 하루 승하차 가운데 몇 %인가
print(len(share), "개 역 ×", share.shape[1], "칸,  가장 붐비는 역:",
      total.idxmax(), round(total.max()))

# 기준선 1: 인원수를 그대로 묶으면?
k_raw = KMeans(n_clusters=3, n_init=10, random_state=42).fit(prof)
print("인원수 그대로 묶은 군집의 평균 하루 승하차:",
      total.groupby(k_raw.labels_).agg(["size", "mean"]).round(0).astype(int).values.tolist())
# 기준선 2: 무리가 없는 데이터 (시간대 열마다 따로 섞기)
Xm = share.to_numpy()
rng2 = np.random.default_rng(0)
Sm = np.column_stack([rng2.permutation(Xm[:, j]) for j in range(Xm.shape[1])])
for k in range(2, 7):
    m1 = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xm)
    m2 = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Sm)
    print(f"k={k}: 실제 실루엣 {silhouette_score(Xm, m1.labels_):.3f}, "
          f"섞은 데이터 {silhouette_score(Sm, m2.labels_):.3f}")

# ### MP-3
K = 3
kmm = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Xm)
cent = pd.DataFrame(kmm.cluster_centers_, columns=share.columns)
am = cent[["승차_07-08시", "승차_08-09시"]].sum(axis=1) - cent[["하차_07-08시", "하차_08-09시"]].sum(axis=1)
print("아침 7~9시 (승차 % − 하차 %):", am.round(1).tolist())
share["유형"] = kmm.labels_
for c in range(K):
    names = share.index[share["유형"] == c]
    top = total[names].sort_values(ascending=False).head(5).index.tolist()
    print(f"유형 {c}: {len(names)}개 역, 붐비는 역 {top}")

fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for c, col in zip(range(K), ["#9a9a9a", "#E04B3C", "#222222"]):
    ax[0].plot(cent.iloc[c, :len(H)].values, color=col, label=f"유형 {c}")
    ax[1].plot(cent.iloc[c, len(H):].values, color=col)
for a_, t in zip(ax, ["승차", "하차"]):
    a_.set_title(t)
    a_.set_xticks([2, 8, 13, 18], ["07시", "13시", "18시", "23시"])
ax[0].set_ylabel("하루 가운데 %")
ax[0].legend(loc="upper center")
plt.show()

# ### MP-4
# 군집에 쓰지 않은 정보로 확인하기: 쉬는 날 승하차 ÷ 평일 승하차
offday = sw[(sw["날종류"] == "쉬는날") & (sw["월"] != 4)]
days = offday.groupby("역명")["날수"].sum() / 2               # 승차·하차 두 줄에 날수가 같이 적혀 있다
off = offday.assign(t=offday[H].sum(axis=1) * offday["날수"]).groupby("역명")["t"].sum() / days
ratio = (off[share.index] / total).rename("쉬는날÷평일")
print(ratio.groupby(share["유형"]).describe()[["count", "25%", "50%", "75%"]].round(2).to_string())

# ### EX-6
db2 = DBSCAN(eps=200 / 6371, min_samples=15,
             metric="haversine", algorithm="ball_tree").fit(rad)
lab = db2.labels_
sizes = pd.Series(lab[lab >= 0]).value_counts()
print("밀집 지대:", len(sizes), "곳, 가장 큰 지대:", sizes.iloc[0], "건,",
      "노이즈:", round((lab == -1).mean() * 100, 1), "%")
print("가장 큰 지대의 지역:", big["region"][lab == sizes.index[0]].value_counts().head(4).to_dict())
