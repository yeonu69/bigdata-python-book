# 11장. 리뷰 속 마음 읽기: 텍스트 분석과 언어 AI (개정판) — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 11-k' 조각 하나가 책의 [파이썬 실습 11-k] 하나입니다. MP-k는 [미니 프로젝트 11]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: PYTHONHASHSEED=0 python3 practice_v2.py
# 필요 라이브러리: pandas, numpy, scikit-learn, matplotlib, kiwipiepy, gensim, pyreadr
#   (코랩에서는 먼저 !pip install kiwipiepy gensim pyreadr koreanize-matplotlib)

# ### 11-1
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

BASE = "https://raw.githubusercontent.com/e9t/nsmc/master/"
train0 = pd.read_csv(BASE + "ratings_train.txt", sep="\t", quoting=3)
test0 = pd.read_csv(BASE + "ratings_test.txt", sep="\t", quoting=3)
print("훈련", train0.shape, " 시험", test0.shape)
print(train0.head(3).to_string())
print("빈 리뷰:", train0["document"].isna().sum(), "/", test0["document"].isna().sum())

train = train0.dropna(subset=["document"]).copy()           # 빈 리뷰 지우기
test = test0.dropna(subset=["document"]).copy()
n_label = train.groupby("document")["label"].nunique()      # 같은 글에 붙은 라벨 종류 수
conflict = n_label[n_label > 1].index                       # 2종류 = 긍정과 부정이 모두 붙은 글
print("같은 글이 다시 나온 횟수:", train["document"].duplicated().sum())
print("같은 글인데 라벨이 엇갈린 글:", len(conflict))
train = train[~train["document"].isin(conflict)].drop_duplicates("document")
print("정리 후 훈련:", len(train), " 시험:", len(test))
y_tr, y_te = train["label"].values, test["label"].values
print("시험 리뷰의 긍정 비율:", round(y_te.mean(), 3))
print("기준선(많은 쪽으로 찍기):", round(max(y_te.mean(), 1 - y_te.mean()), 3))

top = train0[train0["document"].isin(["굿", "good", "최고", "재밌다"])]
cnt = pd.crosstab(top["document"], top["label"])             # 글마다 부정(0), 긍정(1) 개수
print(cnt.to_string())
cnt.plot.barh(stacked=True, color=["#C8234B", "#BBBBBB"], figsize=(6, 3))
plt.legend(["부정 라벨", "긍정 라벨"])
plt.xlabel("훈련 파일에 나온 횟수")
plt.ylabel("")
plt.show()

# ### 11-2
L = train["document"].str.len()
print("길이 평균", round(L.mean(), 1), " 가운데", int(L.median()), " 최대", L.max(),
      " 10자 이하", round((L <= 10).mean(), 3))
print("라벨별 평균 길이:", L.groupby(train["label"]).mean().round(1).to_dict())
rate = {}
for pat in ["최고", "!", "ㅠㅠ|ㅜㅜ", "ㅋㅋ", r"\.\.\.", r"\?", "ㅡㅡ", "쓰레기"]:
    has = train["document"].str.contains(pat)
    key = pat.replace("\\", "")                              # 그래프 이름표에서 역슬래시 빼기
    rate[key] = train.loc[has, "label"].mean()
    print(f"{key:6s} {has.sum():6d}개  긍정 비율 {rate[key]:.3f}")

pd.Series(rate).sort_values().plot.barh(color="#BBBBBB", figsize=(6, 3.5))
plt.axvline(0.5, color="#C8234B", ls="--")
plt.xlabel("이것이 든 리뷰의 긍정 비율")
plt.show()

nxt = pd.Series([t[i + 1] for t in train["document"].str.split()      # "시간" 바로 다음에 온 단어
                 for i in range(len(t) - 1) if t[i] == "시간"])
print('"시간" 다음 단어', len(nxt), "개:", nxt.value_counts().head(5).to_dict())

# ### 11-3
from kiwipiepy import Kiwi

kiwi = Kiwi()
ex = "이 영화 진짜 재미없지는 않았어요ㅋㅋ 배우들 연기가 최고!"
print(ex.split())
print([(t.form, t.tag) for t in kiwi.tokenize(ex)])
print([t.form for t in kiwi.tokenize("너무재밓었다그래서보는것을추천한다")])

def morphs(texts):                                           # 문장 목록 -> "형태소 형태소 ..." 목록
    return [" ".join(t.form for t in toks) for toks in kiwi.tokenize(texts)]

train["m"] = morphs(train["document"].tolist())              # 1~2분 걸립니다
test["m"] = morphs(test["document"].tolist())
print(train[["document", "m"]].head(2).to_string())

ws = train["document"].str.split().explode()                 # 띄어쓰기 단어
ms = train["m"].str.split().explode()                        # 형태소
print("띄어쓰기 단어 종류:", ws.nunique(), " 형태소 종류:", ms.nunique())
print("한 번만 나온 종류의 비율:", round((ws.value_counts() == 1).mean(), 3), round((ms.value_counts() == 1).mean(), 3))
yh = ws[ws.str.fullmatch(r"영화[가-힣]{0,3}")].value_counts()
print("'영화'로 시작하는 단어 꼴:", len(yh), yh.head(5).to_dict())

kinds = pd.Series({"띄어쓰기 단어": ws.nunique(), "형태소": ms.nunique()})
kinds.plot.bar(color=["#BBBBBB", "#C8234B"], figsize=(4, 3))
plt.ylabel("종류 수")
plt.xticks(rotation=0)
plt.show()

# ### 11-4
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = ["배우 연기 최고", "배우 연기 지루", "스토리 지루 지루"]
cv = CountVectorizer(token_pattern=r"\S+")
X = cv.fit_transform(docs).toarray()                         # 문서-단어 행렬(개수)
print(cv.get_feature_names_out())
print(X)
idf = np.log(len(docs) / (X > 0).sum(axis=0))                # idf = ln(N / df)
print("idf:", idf.round(3))
T = X * idf                                                  # TF-IDF = 개수 × idf
print(T.round(3))
print("d1·d2 =", round(T[0] @ T[1], 3), " 길이:", np.linalg.norm(T, axis=1).round(3))
S = cosine_similarity(T)
print("코사인 유사도:\n", S.round(3))

plt.figure(figsize=(4, 3.4))
plt.imshow(S, cmap="Greys", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{S[i, j]:.3f}", ha="center", va="center", color="#C8234B")
plt.xticks(range(3), ["d1", "d2", "d3"])
plt.yticks(range(3), ["d1", "d2", "d3"])
plt.show()

# ### M-1
N, df = 3, np.array([2, 1])                  # "배우"는 리뷰 2개에, "최고"는 1개에 나온다
print("idf:", np.log(N / df).round(3))       # np.log는 자연로그 ln
print("ln 10, ln 100:", np.log([10, 100]).round(3))
print("사이킷런 idf:", (np.log((1 + N) / (1 + df)) + 1).round(3))

# ### M-2
a, b = np.log(3 / 2), np.log(3)               # idf: 리뷰 2개에 나온 단어, 1개에 나온 단어
d1 = np.array([a, 0, a, 0, b])               # (배우, 스토리, 연기, 지루, 최고)
d2 = np.array([a, 0, a, a, 0])
dot = d1 @ d2                                # 내적: 같은 칸끼리 곱해 모두 더하기
n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
print("내적", round(dot, 3), " 길이", round(n1, 3), round(n2, 3), " 코사인", round(dot / (n1 * n2), 3))

# ### M-3
u = np.array([0, 1, 2, 3])
w = u + 100                                  # 모두 100씩 크다
cos = u @ w / (np.linalg.norm(u) * np.linalg.norm(w))
print("코사인:", round(cos, 3), " 상관계수:", round(np.corrcoef(u, w)[0, 1], 3))

# ### 11-5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def fit_score(col, **kw):                                    # 자르는 방법을 받아 TF-IDF + 로지스틱 회귀
    v = TfidfVectorizer(lowercase=False, sublinear_tf=True, **kw)
    m = LogisticRegression(C=3, max_iter=3000).fit(v.fit_transform(train[col]), y_tr)
    return v, m, m.predict(v.transform(test[col]))

v_w, lr_w, pred_w = fit_score("document", token_pattern=r"\S+")
print("띄어쓰기 단어  정확도:", round((pred_w == y_te).mean(), 3))
vec, lr, pred = fit_score("m", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2)
print("형태소 1~2개씩 정확도:", round((pred == y_te).mean(), 3), " 조각 수:", len(vec.vocabulary_),
      " 리뷰당 평균:", round(vec.transform(train["m"]).getnnz(axis=1).mean(), 1))
print(confusion_matrix(y_te, pred))
print("정밀도:", round(precision_score(y_te, pred), 3), " 재현율:", round(recall_score(y_te, pred), 3))
ov = test["document"].isin(train0["document"]).values        # 훈련 파일에도 똑같은 글이 있는 시험 리뷰
print("겹친 리뷰", ov.sum(), "개 정확도", round((pred[ov] == y_te[ov]).mean(), 3),
      " 나머지", round((pred[~ov] == y_te[~ov]).mean(), 3))

acc = pd.Series({"찍기": max(y_te.mean(), 1 - y_te.mean()), "띄어쓰기": (pred_w == y_te).mean(),
                 "형태소 1~2개": (pred == y_te).mean()})
acc.plot.bar(color=["#DDDDDD", "#BBBBBB", "#C8234B"], figsize=(4.5, 3))
plt.ylim(0.45, 0.9)
plt.ylabel("시험 정확도")
plt.xticks(rotation=0)
plt.show()

# ### CV-1
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv5 = StratifiedKFold(5, shuffle=True, random_state=0)       # 훈련 리뷰만 다섯 조각으로 (시험 리뷰는 쓰지 않는다)
for c in [0.3, 1, 3, 10]:
    pipe = make_pipeline(TfidfVectorizer(lowercase=False, sublinear_tf=True, token_pattern=r"\S+",
                                         ngram_range=(1, 2), min_df=2),
                         LogisticRegression(C=c, max_iter=3000))
    s = cross_val_score(pipe, train["m"], y_tr, cv=cv5, n_jobs=-1)   # 조각마다 TF-IDF 사전부터 새로 배운다
    print(f"C={c:<4} 검증 정확도 {s.mean():.4f}")

# ### 11-6
from sklearn.naive_bayes import MultinomialNB

ways = {"형태소 1개씩": ("m", dict(token_pattern=r"\S+")),
        "글자 1~3개씩": ("document", dict(analyzer="char", ngram_range=(1, 3), min_df=2))}
res = {"띄어쓰기 단어": (pred_w == y_te).mean()}
for name, (col, kw) in ways.items():
    v, m, p = fit_score(col, **kw)
    res[name] = (p == y_te).mean()
    print(f"{name}  조각 수 {len(v.vocabulary_):7d}  정확도 {res[name]:.3f}")
res["형태소 1~2개씩"] = (pred == y_te).mean()

cvm = CountVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, lowercase=False)
nb = MultinomialNB().fit(cvm.fit_transform(train["m"]), y_tr)
acc_nb = (nb.predict(cvm.transform(test["m"])) == y_te).mean()
print("나이브 베이즈(형태소 1~2개, 개수) 정확도:", round(acc_nb, 3))

pd.Series(res).plot.barh(color="#BBBBBB", figsize=(6, 3))
plt.axvline(acc_nb, color="#C8234B", ls="--", label="나이브 베이즈")
plt.xlim(0.75, 0.9)
plt.xlabel("시험 정확도 (로지스틱 회귀)")
plt.legend()
plt.show()

# ### 11-7
names = vec.get_feature_names_out()
coef = lr.coef_[0]
order = np.argsort(coef)                                     # 가중치가 작은 것부터의 순서
print("긍정 쪽:", [(names[i], round(float(coef[i]), 1)) for i in order[::-1][:8]])
print("부정 쪽:", [(names[i], round(float(coef[i]), 1)) for i in order[:8]])

new = ["배우들 연기가 정말 최고였어요", "시간이 아까운 영화", "재미없을 줄 알았는데 의외로 재밌네요",
       "기대 안 했는데 대박", "재미없지는 않았다", "그냥 그랬어요"]
p_new = lr.predict_proba(vec.transform(morphs(new)))[:, 1]
for s, p in zip(new, p_new):
    print(f"긍정 확률 {p:.3f}  {s}")

pick = np.r_[order[:6], order[-6:]]
plt.figure(figsize=(6, 4))
plt.barh(range(12), coef[pick], color=["#BBBBBB"] * 6 + ["#C8234B"] * 6)
plt.yticks(range(12), names[pick])
plt.xlabel("가중치 (왼쪽 = 부정, 오른쪽 = 긍정)")
plt.show()

# ### M-4
x = vec.transform(morphs(["재미없지는 않았다"]))            # 이 리뷰의 TF-IDF 무게(0이 아닌 칸만)
part = pd.Series(coef[x.indices] * x.data, index=names[x.indices]).sort_values()   # 가중치 × 무게
print(part.round(3).to_dict())
print("절편 + 합 =", round(lr.intercept_[0] + part.sum(), 3))

# ### 11-8
prob = lr.predict_proba(vec.transform(test["m"]))[:, 1]       # 시험 리뷰마다 긍정 확률
sure = (prob >= 0.9) | (prob <= 0.1)                          # 확신하는 답
unsure = (prob > 0.4) & (prob < 0.6)                          # 망설이는 답
print(f"확신: 리뷰의 {sure.mean():.3f}, 정확도 {(pred[sure] == y_te[sure]).mean():.3f}")
print(f"망설임: 리뷰의 {unsure.mean():.3f}, 정확도 {(pred[unsure] == y_te[unsure]).mean():.3f}")
print("틀린 리뷰:", (pred != y_te).sum(), "개")

wrong = test.assign(prob=prob)[pred != y_te].drop_duplicates("document")
short = wrong[wrong["document"].str.len() <= 12]
print(short[short["label"] == 0].nlargest(4, "prob")[["document", "prob"]].round(3).to_string(index=False))
print(short[short["label"] == 1].nsmallest(4, "prob")[["document", "prob"]].round(3).to_string(index=False))

plt.figure(figsize=(6, 3))
plt.hist([prob[y_te == 0], prob[y_te == 1]], bins=20, stacked=True,
         color=["#BBBBBB", "#C8234B"], label=["실제 부정", "실제 긍정"])
plt.xlabel("모델이 낸 긍정 확률")
plt.ylabel("시험 리뷰 수")
plt.legend()
plt.show()

# ### 11-9
rng = np.random.RandomState(0)
perm = rng.permutation(len(train))                           # 훈련 리뷰를 섞은 순서
curve = {}
for n in [1000, 5000, 20000, 50000]:
    idx = perm[:n]
    v = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2 if n > 5000 else 1,
                        lowercase=False, sublinear_tf=True)
    m = LogisticRegression(C=3, max_iter=3000).fit(v.fit_transform(train["m"].values[idx]), y_tr[idx])
    curve[n] = (m.predict(v.transform(test["m"])) == y_te).mean()
curve[len(train)] = (pred == y_te).mean()
for n, a in curve.items():
    print(f"훈련 리뷰 {n:6d}개  정확도 {a:.3f}")

pd.Series(curve).plot(marker="o", logx=True, color="#C8234B", figsize=(5, 3))
plt.xlabel("훈련 리뷰 수 (눈금은 10배씩)")
plt.ylabel("시험 정확도")
plt.show()

# ### 11-10
from gensim.models import Word2Vec

v1 = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True).fit(train["m"])
for a, b in [("연기 최고", "연기력 훌륭"), ("연기 최고", "연기 최악")]:
    s = cosine_similarity(v1.transform(morphs([a])), v1.transform(morphs([b])))[0, 0]
    print(f"TF-IDF 코사인  {a} / {b}: {s:.3f}")

sents = train["m"].str.split().tolist()
w2v = Word2Vec(sents, vector_size=100, window=5, min_count=5, sg=1, epochs=5, seed=42, workers=1)
for w in ["배우", "지루", "최고"]:
    print(w, "->", [(x, round(float(s), 2)) for x, s in w2v.wv.most_similar(w, topn=4)])
for a, b in [("지루", "따분"), ("연기", "연기력"), ("최고", "최악")]:
    print(f"Word2Vec 유사도  {a} / {b}: {w2v.wv.similarity(a, b):.3f}")

def doc_vec(m):                                              # 리뷰 = 단어 벡터의 평균
    vs = [w2v.wv[w] for w in m.split() if w in w2v.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(100)
Xe = np.array([doc_vec(m) for m in train["m"]])
Te = np.array([doc_vec(m) for m in test["m"]])
lre = LogisticRegression(max_iter=3000).fit(Xe, y_tr)
print("평균 벡터 + 로지스틱 회귀 정확도:", round((lre.predict(Te) == y_te).mean(), 3))
print("왕 - 남자 + 여자 ->", [x for x, _ in w2v.wv.most_similar(positive=["왕", "여자"], negative=["남자"], topn=3)])

q = "배우들 연기가 정말 좋았다"                              # 가장 비슷한 리뷰 찾기
sim = cosine_similarity(v1.transform(morphs([q])), v1.transform(train["m"]))[0]
for i in np.argsort(-sim)[:3]:
    print(f"{sim[i]:.3f}  {train['document'].values[i]}")

# ### 11-11
import re
import urllib.request
import warnings
import pyreadr
warnings.filterwarnings("ignore", category=RuntimeWarning)   # 날짜 칸을 읽을 때 나는 경고 숨기기

urllib.request.urlretrieve("https://raw.githubusercontent.com/cran/corpus/master/data/federalist.rda",
                           "federalist.rda")
fed = pyreadr.read_r("federalist.rda")["federalist"]
fed["no"] = fed["name"].str.extract(r"(\d+)").astype(int)
DISP = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63]      # 저자 논란이 있는 12편
fed["저자"] = np.where(fed["no"].isin(DISP), "논란", fed["author"].fillna("공저"))
print(fed["저자"].value_counts().to_dict())

def per1000(texts, word):                                    # 1,000단어당 횟수
    words = re.findall(r"[a-z]+", " ".join(texts).lower())
    return 1000 * words.count(word) / len(words)
fw = pd.DataFrame({g: [per1000(fed.loc[fed["저자"] == g, "text"], w) for w in ["upon", "whilst", "on"]]
                   for g in ["Hamilton", "Madison", "논란"]}, index=["upon", "whilst", "on"])
print(fw.round(2).to_string())
print("upon이 없는 논란의 글:", sum(per1000([t], "upon") == 0 for t in fed.loc[fed["저자"] == "논란", "text"]), "편")

FW = ["upon", "whilst", "while", "on", "by", "there", "although", "also", "enough", "an", "of",
      "to", "would", "may", "this", "the", "and", "in", "a", "is", "it", "that", "which", "be",
      "as", "with", "for", "from", "at", "or", "not"]           # 내용과 상관없는 기능어 31개
cvf = CountVectorizer(vocabulary=FW, token_pattern=r"[a-z]+")
known = fed[fed["저자"].isin(["Hamilton", "Madison"])]
nbf = MultinomialNB().fit(cvf.transform(known["text"]), known["저자"])
dd = fed[fed["저자"] == "논란"].sort_values("no")
print(dict(zip(dd["no"].tolist(), nbf.predict(cvf.transform(dd["text"])).tolist())))

fw.plot.bar(color=["#BBBBBB", "#C8234B", "#555555"], figsize=(5.5, 3))
plt.ylabel("1,000단어당 횟수")
plt.xticks(rotation=0)
plt.show()

lam = fw.loc["upon", ["Hamilton", "Madison"]] * 2           # 2,000단어 글에서 기대하는 "upon" 횟수
p0 = np.exp(-lam)                                            # 한 번도 안 나올 확률 e^(-λ)
print("기대 횟수", lam.round(2).to_dict(), " 0번일 확률", p0.round(5).to_dict())
print("매디슨 쪽으로", round(p0["Madison"] / p0["Hamilton"]), "배")

# ### MP-1
URL = "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt"
shop = pd.read_csv(URL, sep="\t", header=None, names=["별점", "리뷰"], quoting=3)
print(shop.shape)
print(shop["별점"].value_counts().sort_index().to_dict())
shop["label"] = (shop["별점"] >= 4).astype(int)              # 4~5점 긍정, 1~2점 부정
n_lab = shop.groupby("리뷰")["label"].nunique()
shop = shop[~shop["리뷰"].isin(n_lab[n_lab > 1].index)].drop_duplicates("리뷰")
print("정리 후:", len(shop), " 긍정 비율:", round(shop["label"].mean(), 3))
print("길이 가운데 값:", int(shop["리뷰"].str.len().median()), "자")

shop["별점"].value_counts().sort_index().plot.bar(color="#BBBBBB", figsize=(4, 3))
plt.xlabel("별점")
plt.ylabel("리뷰 수")
plt.xticks(rotation=0)
plt.show()

# ### MP-2
from sklearn.model_selection import train_test_split

s_tr, s_te = train_test_split(shop, test_size=0.25, random_state=42, stratify=shop["label"])
s_tr, s_te = s_tr.copy(), s_te.copy()
s_tr["m"] = morphs(s_tr["리뷰"].tolist())
s_te["m"] = morphs(s_te["리뷰"].tolist())
ys = s_te["label"].values
print("시험 리뷰:", len(s_te), " 기준선:", round(max(ys.mean(), 1 - ys.mean()), 3))
p_movie = lr.predict(vec.transform(s_te["m"]))               # 영화 리뷰로 배운 모델 그대로
print("영화 모델의 정확도:", round((p_movie == ys).mean(), 3))
print(confusion_matrix(ys, p_movie))

# ### MP-3
vs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, lowercase=False, sublinear_tf=True)
lrs = LogisticRegression(C=3, max_iter=3000).fit(vs.fit_transform(s_tr["m"]), s_tr["label"])
p_shop = lrs.predict(vs.transform(s_te["m"]))
print("쇼핑 모델의 정확도:", round((p_shop == ys).mean(), 3))

few = s_tr.sample(1000, random_state=0)                      # 쇼핑 리뷰 1,000개만
v_few = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False, sublinear_tf=True)
m_few = LogisticRegression(C=3, max_iter=3000).fit(v_few.fit_transform(few["m"]), few["label"])
acc_few = (m_few.predict(v_few.transform(s_te["m"])) == ys).mean()
print("쇼핑 리뷰 1,000개로 배운 모델:", round(acc_few, 3))

board = pd.Series({"찍기": 0.5, "영화 리뷰\n14만 개": (p_movie == ys).mean(),
                   "쇼핑 리뷰\n1,000개": acc_few, "쇼핑 리뷰\n15만 개": (p_shop == ys).mean()})
board.plot.bar(color=["#DDDDDD", "#BBBBBB", "#C8234B", "#C8234B"], figsize=(5.5, 3))
plt.ylim(0.45, 1)
plt.ylabel("쇼핑 리뷰 시험 정확도")
plt.xticks(rotation=0)
plt.show()

# ### MP-4
for w in ["배송", "사이즈", "반품", "재구매"]:
    has = s_te["리뷰"].str.contains(w).values
    print(f"{w:4s} {has.sum():5d}개  영화 모델 {(p_movie[has] == ys[has]).mean():.3f}"
          f"  쇼핑 모델 {(p_shop[has] == ys[has]).mean():.3f}")
probe = ["배송이 빨라요", "생각보다 작아요", "반품했어요", "재구매 의사 있어요"]
pm = lr.predict_proba(vec.transform(morphs(probe)))[:, 1]
ps = lrs.predict_proba(vs.transform(morphs(probe)))[:, 1]
for s, a, b in zip(probe, pm, ps):
    print(f"영화 모델 {a:.3f}  쇼핑 모델 {b:.3f}  {s}")

# ### EX-6
hand = (prob > 0.3) & (prob < 0.7)                           # 사람에게 넘길 리뷰
print("사람에게 넘기는 리뷰:", hand.sum(), "개, 비율", round(hand.mean(), 3))
print("모델이 맡은 리뷰의 정확도:", round((pred[~hand] == y_te[~hand]).mean(), 3))
print("넘긴 리뷰를 모델이 판정했다면:", round((pred[hand] == y_te[hand]).mean(), 3))
