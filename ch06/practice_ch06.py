# 06장. 파일에서, 웹에서: 데이터 불러오기 — 05절 파이썬 실습과 미니 프로젝트 전체 코드
# '# ### 6-k' 조각 하나가 책의 [파이썬 실습 6-k] 하나입니다. MP-k는 [미니 프로젝트 6]의 단계, EX-k는 연습문제 정답 코드입니다.
# 준비: 강의 자료 「6장_강의자료.zip」을 코랩 왼쪽 파일 창에 올리고 압축을 풉니다(!unzip -q 6장_강의자료.zip).
# 필요 라이브러리: pandas, openpyxl, requests, beautifulsoup4, lxml, pyarrow, matplotlib (코랩에는 모두 들어 있습니다)

# ### 6-1
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

DATA = "6장_강의자료"
PATH = DATA + "/기상청_서울_시간자료_원본.csv"   # 기상자료개방포털에서 받은 그대로

try:
    w = pd.read_csv(PATH)                        # 아무것도 정하지 않으면 UTF-8로 읽는다
except UnicodeDecodeError as e:
    print("오류:", e)

w = pd.read_csv(PATH, encoding="cp949")          # 한글 윈도의 옛 방식으로 다시 읽기
print("크기:", w.shape)
print(w.iloc[:3, [1, 2, 3, 11]].to_string())

for enc in ["utf-8", "cp949"]:                   # 같은 글자, 다른 바이트
    b = "서울".encode(enc)
    print(f"{enc:>6}: {len(b)}바이트 {b.hex(' ')}")
print("파일 맨 앞 4바이트:", open(PATH, "rb").read(4).hex(" "), "→", open(PATH, "rb").read(4).decode("cp949"))

# ### 6-2
import time
f = DATA + "/에어코리아_서울_2025/2025년 1월.xlsx"
t0 = time.time()
one = pd.read_excel(f)
print("읽는 데 걸린 시간:", round(time.time() - t0, 1), "초")
print("크기:", one.shape)
print(one.dtypes.to_string())
print(one.head(3).to_string())
print(one["망"].value_counts().to_dict())
print("측정소 수:", one["측정소명"].nunique())
print("측정일시의 시:", one["측정일시"].min() % 100, "~", one["측정일시"].max() % 100)

try:
    pd.to_datetime(one["측정일시"].astype(str), format="%Y%m%d%H")
except ValueError as e:
    print("날짜로 바꾸지 못함:", str(e).split(". You")[0])

day = pd.to_datetime((one["측정일시"] // 100).astype(str), format="%Y%m%d")   # 앞 8자리 = 날짜
one["일시"] = day + pd.to_timedelta(one["측정일시"] % 100, unit="h")          # 뒤 2자리 = 시
print(one[["측정일시", "일시"]].iloc[22:26].to_string())

# ### 6-3
import glob, os, re
files = glob.glob(DATA + "/에어코리아_서울_2025/*.xlsx")
print("파일 수:", len(files))
print("글자 순서:", [os.path.basename(f) for f in sorted(files)][:4])

def month_of(path):                              # 파일 이름에서 '숫자+월'을 찾아 숫자만
    return int(re.search(r"(\d+)월", path).group(1))

files = sorted(files, key=month_of)
print("달 순서:", [os.path.basename(f) for f in files][:4])
print("빠진 달:", sorted(set(range(1, 13)) - {month_of(f) for f in files}))

parts = []
for f in files:
    t = pd.read_excel(f)
    t["파일"] = os.path.basename(f)              # 어느 파일에서 왔는지 표시
    parts.append(t)
print("열 이름이 모두 같은가:", all(list(p.columns) == list(parts[0].columns) for p in parts))

air = pd.concat(parts, ignore_index=True)
print("합친 크기:", air.shape)
print("행 수 검산:", sum(len(p) for p in parts), "=", len(air))

air["달"] = air["측정일시"] // 10000 % 100
days = air.groupby("달")["측정일시"].apply(lambda s: (s // 100).nunique())
check = pd.DataFrame({"행": air.groupby("달").size(), "기대": days * 24 * air["측정소코드"].nunique()})
print(check.T.to_string())

city = air[air["망"] == "도시대기"]
city.groupby("달")["PM25"].mean().plot.bar(color="#8a8a8a", figsize=(7, 3))
plt.xlabel("달 (2025년)")
plt.ylabel("PM2.5 평균")
plt.xticks(rotation=0)
plt.show()

# ### 6-4
import subprocess, sys, time
import requests

SITE = DATA + "/연습사이트"
server = subprocess.Popen([sys.executable, "serve.py", "8000"], cwd=SITE)   # 연습 사이트 켜기
time.sleep(2)
BASE = "http://localhost:8000"

r = requests.get(BASE + "/list/1.html")
print("상태 코드:", r.status_code)
print("기본 User-Agent:", requests.utils.default_user_agent())

HEAD = {"User-Agent": "data-class-bot/1.0 (book ch6 practice)"}   # 누구인지, 왜 왔는지 (영문만)
r = requests.get(BASE + "/list/1.html", headers=HEAD)
print("상태 코드:", r.status_code, "|", r.headers["Content-Type"], "|", len(r.text), "글자")
print(r.text[:120])
print("없는 쪽:", requests.get(BASE + "/list/9.html", headers=HEAD).status_code)

from urllib import robotparser
rp = robotparser.RobotFileParser(BASE + "/robots.txt")
rp.read()
for path in ["/list/1.html", "/station/111121.html", "/admin/index.html"]:
    print(f"{path:22} 가져가도 되나: {rp.can_fetch(HEAD['User-Agent'], BASE + path)}")
print("요청 사이에 쉴 시간:", rp.crawl_delay("*"), "초")

# ### 6-5
from bs4 import BeautifulSoup

soup = BeautifulSoup(r.text, "html.parser")       # 글자 → 나무 구조
print(soup.title.text)
print(soup.find("h1").text)

rows = soup.find("table", class_="stations").find("tbody").find_all("tr")
print("줄 수:", len(rows))
print(rows[0])

recs = []
for tr in rows:
    td = tr.find_all("td")
    a = td[1].find("a")
    recs.append({"구": td[0].text, "측정소": a.text, "코드": td[2].text, "링크": a["href"]})
page1 = pd.DataFrame(recs)
print(page1.head(3).to_string())
print("다음 쪽 링크:", soup.find("a", class_="next")["href"])

# ### 6-6
from urllib.parse import urljoin

def get_soup(url):
    time.sleep(rp.crawl_delay("*"))               # robots.txt의 약속대로 쉬었다가
    res = requests.get(url, headers=HEAD)
    res.raise_for_status()                        # 200이 아니면 여기서 멈춘다
    return BeautifulSoup(res.text, "html.parser")

stations, url = [], BASE + "/list/1.html"
while url:                                        # '다음' 링크가 없을 때까지
    s = get_soup(url)
    for tr in s.find("table", class_="stations").find("tbody").find_all("tr"):
        td = tr.find_all("td")
        stations.append({"구": td[0].text, "측정소": td[1].text, "코드": td[2].text,
                         "주소": urljoin(url, td[1].find("a")["href"])})
    nxt = s.find("a", class_="next")
    url = urljoin(url, nxt["href"]) if nxt else None
st = pd.DataFrame(stations)
print("측정소:", len(st), "곳 | 첫 주소:", st["주소"][0])

t0, rows = time.time(), []
for one_st in st.itertuples():
    s = get_soup(one_st.주소)
    for tr in s.find("table", class_="monthly").find("tbody").find_all("tr"):
        td = [x.text for x in tr.find_all("td")]
        rows.append({"코드": one_st.코드, "구": one_st.구, "달": int(td[0][5:]),
                     "PM25": pd.to_numeric(td[2], errors="coerce"),   # '-'는 빈칸(NaN)으로
                     "시간": int(td[3])})
web = pd.DataFrame(rows)
print("웹에서 모은 표:", web.shape, "| 걸린 시간:", round(time.time() - t0), "초")
print(web[web["PM25"].isna()].to_string(index=False))

# 파일 길(6-3)과 웹 길이 같은 답을 주는지 맞춰 보기
fm = (city.groupby(["측정소코드", "달"])["PM25"].mean().round(1)
          .reset_index().rename(columns={"측정소코드": "코드", "PM25": "파일"}))
fm["코드"] = fm["코드"].astype(str)
both = web.merge(fm, on=["코드", "달"])
same = (both["PM25"] == both["파일"]) | (both["PM25"].isna() & both["파일"].isna())
print("맞춰 본 칸:", len(both), "| 같은 칸:", same.sum(), "(둘 다 빈칸", (both["PM25"].isna() & both["파일"].isna()).sum(), "개 포함)")

gu = web.groupby("구")["PM25"].mean().sort_values()
gu.plot.barh(color="#8a8a8a", figsize=(6, 7))
plt.xlabel("2025년 PM2.5 (달 평균 12개의 평균)")
plt.ylabel("")
plt.show()
print("가장 낮은 구:", gu.index[0], gu.iloc[0].round(1), "| 가장 높은 구:", gu.index[-1], gu.iloc[-1].round(1))

# ### 6-7
import io
html = requests.get(st["주소"][0], headers=HEAD).text
tables = pd.read_html(io.StringIO(html))          # 페이지 속 <table>을 모두 표로
print("표 개수:", len(tables))
print(tables[0].head(3).to_string())

# ### 6-8
s = get_soup(st["주소"][0])
box = s.find("div", id="daily")
print("파이썬이 받은 칸:", box.text)
print("칸이 부르는 자료:", box["data-src"])

js = requests.get(urljoin(st["주소"][0], box["data-src"]), headers=HEAD).json()
print("JSON 열쇠:", list(js.keys()))
print(js["daily"][:2])
d = pd.DataFrame(js["daily"])
d["date"] = pd.to_datetime(d["date"])
print(d.shape, "| 가장 높은 날:", d.loc[d["pm25"].idxmax(), "date"].date(), d["pm25"].max())
d.plot(x="date", y="pm25", color="#C8234B", legend=False, figsize=(8, 3))
plt.xlabel("")
plt.ylabel("하루 평균 PM2.5")
plt.show()

# ### 6-9
os.makedirs("저장", exist_ok=True)
keep = air.drop(columns=["파일"])
for kind in ["csv", "parquet", "xlsx"]:
    path = "저장/서울_2025." + kind
    t0 = time.time()
    if kind == "csv": keep.to_csv(path, index=False)
    elif kind == "parquet": keep.to_parquet(path, index=False)
    else: keep.to_excel(path, index=False)
    t_w = time.time() - t0
    t0 = time.time()
    back = {"csv": pd.read_csv, "parquet": pd.read_parquet, "xlsx": pd.read_excel}[kind](path)
    print(f"{kind:>7}: {os.path.getsize(path) / 1e6:5.1f} MB | 쓰기 {t_w:4.1f}초 | 읽기 {time.time() - t0:4.1f}초 | {back.shape}")

n_bad = 0
for chunk in pd.read_csv("저장/서울_2025.csv", chunksize=100_000):   # 10만 줄씩 나눠 읽기
    n_bad += (chunk["PM25"] > 75).sum()
print("PM2.5가 75를 넘은 측정:", n_bad, "번")

# ### 6-10
import unicodedata
name = "2025년 1월.xlsx"
mac = unicodedata.normalize("NFD", name)          # 맥에서 압축한 파일 이름의 모양
print(name == mac, len(name), len(mac))
print([hex(ord(c)) for c in mac[4:7]])
print("다시 합치면:", unicodedata.normalize("NFC", mac) == name)

server.terminate()                                 # 연습 사이트 끄기

# ### MP-1
P = DATA + "/전력수요/한국전력거래소_시간별_전국_전력수요량_2021.csv"
y21 = pd.read_csv(P, encoding="cp949")
print(y21.shape)
print(y21.iloc[:2, :6].to_string())
print(y21.dtypes.value_counts().to_dict())

# ### MP-2
files = sorted(glob.glob(DATA + "/전력수요/*.csv"))
parts = [pd.read_csv(f, encoding="cp949") for f in files]
print([len(p) for p in parts])
print("열이 모두 같은가:", all(list(p.columns) == list(parts[0].columns) for p in parts))
kpx = pd.concat(parts, ignore_index=True)
print("합친 크기:", kpx.shape, "| 날짜가 빈 행:", kpx["날짜"].isna().sum())
print(kpx[kpx["날짜"].isna()].iloc[:, :4].to_string())
kpx = kpx.dropna(subset=["날짜"])
kpx = kpx.astype({c: "int64" for c in kpx.columns[1:]})   # 빈 행 때문에 실수가 된 열을 정수로
print("빈 행을 지운 뒤:", kpx.shape, "| 같은 날짜:", kpx["날짜"].duplicated().sum(), "| 빈칸:", kpx.isna().sum().sum())

# ### MP-3
long = kpx.melt(id_vars="날짜", var_name="시", value_name="수요")    # 넓은 표 → 긴 표
long["시"] = long["시"].str.replace("시", "").astype(int)
long["일시"] = pd.to_datetime(long["날짜"]) + pd.to_timedelta(long["시"], unit="h")
long = long.sort_values("일시").reset_index(drop=True)
print(long.shape)
print(long.head(3).to_string())
print("시간 간격:", long["일시"].diff().value_counts().to_dict())

# ### MP-4
long["연도"] = long["일시"].dt.year
top = long.loc[long.groupby(long["날짜"].str[:4])["수요"].idxmax(), ["날짜", "시", "수요"]]
print(top.to_string(index=False))
prof = long.pivot_table(index="시", columns=long["날짜"].str[:4], values="수요", aggfunc="mean")
prof.plot(figsize=(7, 3.5), color=["#d0d0d0", "#b0b0b0", "#8a8a8a", "#555555", "#C8234B"])
plt.xlabel("시 (1~24)")
plt.ylabel("평균 전력수요 (MW)")
plt.legend(title="연도", ncol=5, fontsize=9, loc="upper left")
plt.show()

# ### EX-1
for word in ["데이터", "AI"]:
    print(word, len(word.encode("utf-8")), len(word.encode("cp949")))

# ### EX-3
from bs4 import BeautifulSoup
html = """<ul class="news">
  <li><a href="/a1">첫 기사</a></li>
  <li><a href="/a2">둘째 기사</a></li>
</ul>
<p class="date">2025-10-20</p>"""
s = BeautifulSoup(html, "html.parser")
print(s.find("li").text)
print([a["href"] for a in s.find_all("a")])
print(s.find("p", class_="date").text)

# ### EX-4
from urllib import robotparser
rules = """User-agent: GPTBot
Disallow: /

User-agent: *
Disallow: /search/
Crawl-delay: 2"""
rp2 = robotparser.RobotFileParser()
rp2.parse(rules.splitlines())
print(rp2.can_fetch("data-class-bot", "/news/1.html"), rp2.can_fetch("GPTBot", "/news/1.html"), rp2.crawl_delay("*"))

# ### EX-2
print(2025013124 // 100, 2025013124 % 100)
t = pd.to_datetime("20250131", format="%Y%m%d") + pd.to_timedelta(24, unit="h")
print(t)
