# 02장. 숫자를 한꺼번에: 파이썬과 넘파이 (개정판) — 05절 실습, 04절 확인 코드, 미니 프로젝트, 연습문제 정답 코드
# '# ### 2-k' 조각 하나가 책의 [파이썬 실습 2-k] 하나입니다.
# M-k는 04절 손 계산을 파이썬으로 확인하는 짧은 코드, MP-k는 [미니 프로젝트 2]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: numpy, pandas, matplotlib, koreanize-matplotlib, scikit-learn)
# 미니 프로젝트는 강의 자료 파일 kpx_시간별_전국_전력수요량_2021_2025.csv가 같은 폴더에 있어야 합니다.

# ### 2-1
import matplotlib.pyplot as plt
import koreanize_matplotlib     # 그래프의 한글이 깨지지 않게 (코랩에서 한 번 설치)

bikes = 254            # 정수(int): 2017년 12월 1일 0시의 대여량
temp = -5.2            # 실수(float): 그때 기온(°C)
season = "Winter"      # 문자열(str): 계절
print(type(bikes), type(temp), type(season), bikes > 200)
print(3 + 4, "3" + "4")

spend = [4500, 12000, 3200, 8900, 15000, 2500, 6800]   # 월~일 지출(원)
total = 0
big = []
for x in spend:            # spend에서 값을 하나씩 꺼내 x라고 부르며 되풀이
    total = total + x      # 지금까지의 합에 x를 더한다
    if x >= 10000:         # x가 1만 원 이상이면
        big.append(x)      # big 뒤에 붙인다
print("합계:", total, "/ sum():", sum(spend), "/ 평균:", round(total / len(spend), 1))
print("1만 원 이상:", big, "/ 월요일:", spend[0], "/ 일요일:", spend[-1])

plt.bar(["월", "화", "수", "목", "금", "토", "일"], spend)
plt.axhline(10000, color="red", linestyle="--")      # 1만 원 선
plt.title("일주일 지출(원)")
plt.show()

# ### 2-2
import numpy as np

def mean(values):                       # 평균을 내는 함수: 이름 붙여 다시 쓴다
    return sum(values) / len(values)

print("성적 평균:", mean([85, 92, 78]), "/ 지출 평균:", round(mean(spend), 1))

price = np.array([1500, 3200, 4800])    # 우유, 빵, 과일 한 개 값(원)
qty = np.array([4, 2, 3])               # 담은 개수
print(price * qty)                      # 자리끼리 곱하기
print((price * qty).sum(), price @ qty) # 합계 = 내적
print([1500, 3200, 4800] * 2)           # 리스트에 2를 곱하면?
print(price * 2)                        # 배열에 2를 곱하면?

score = np.array([85, 92, 78])          # 세 과목 점수
credit = np.array([3, 2, 1])            # 학점
print("학점 평균:", score @ credit, "÷", credit.sum(), "=", round(score @ credit / credit.sum(), 2))

plt.bar(["우유", "빵", "과일"], price * qty)
plt.title("품목별 금액 = 가격 × 개수 (원)")
plt.show()

# ### 2-3
import pandas as pd

URL = ("https://raw.githubusercontent.com/syedsharin/"
       "Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv")
df = pd.read_csv(URL, encoding="latin-1")
df.columns = ["date", "count", "hour", "temp", "humidity", "wind", "visibility",
              "dew", "solar", "rain", "snow", "season", "holiday", "functioning"]
count = df["count"].to_numpy()          # 시간별 대여량 8,760개
hour = df["hour"].to_numpy()            # 그 행의 시각 0~23
print(type(count), count.shape, count.dtype, count[:5])
print("합계:", count.sum(), "/ 평균:", count.mean().round(1), "/ 최댓값:", count.max())

print("0인 시간:", (count == 0).sum())  # True를 1로 세어 더한다
busy = count >= 3000                    # True/False 8,760개
print("3,000대 이상:", busy.sum(), "시간 / 그 시각:", np.unique(hour[busy]))
i = count.argmax()                      # 가장 큰 값의 자리 번호
print("가장 붐빈 시간:", df["date"][i], hour[i], "시,", count[i], "대")

plt.hist(count, bins=40)
plt.axvline(3000, color="red", linestyle="--")
plt.title("한 시간 대여량 8,760개의 분포")
plt.xlabel("한 시간 대여량(대)")
plt.ylabel("시간 수")
plt.show()

# ### 2-4
print("모든 날이 0~23시 순서인가:", (hour.reshape(365, 24) == np.arange(24)).all())
M = count.reshape(365, 24)              # 24칸씩 접어 365줄로
print(M.shape)
print(M[0])                             # 첫날(2017년 12월 1일) 24시간
print(M[0, 18], M[:3, 8])               # 첫날 18시, 첫 사흘의 8시

days = df["date"].to_numpy()[::24]      # 24칸마다 하나씩: 날마다 날짜 하나
daily = M.sum(axis=1)                   # 옆으로 더하기 → 하루 합계 365개
d = daily.argmax()
print("가장 많은 날:", days[d], daily[d], "대 / 공휴일 열:", df["holiday"][d * 24])
print("공휴일로 적힌 첫 세 날:", days[df["holiday"].to_numpy()[::24] == "Holiday"][:3])
print("합계가 0인 날:", (daily == 0).sum(), "일")

plt.figure(figsize=(7, 4))
plt.imshow(M.T, aspect="auto", cmap="YlOrRd")   # 가로 365일, 세로 24시간
plt.colorbar(label="대여량(대)")
plt.xlabel("2017년 12월 1일부터 며칠째")
plt.ylabel("시각(시)")
plt.show()

# ### 2-5
on = (df["functioning"] == "Yes").to_numpy().reshape(365, 24)
full = on.all(axis=1)                   # 24시간 모두 운영한 날
print("24시간 운영한 날:", full.sum(), "일 / 아닌 날:", (~full).sum(), "일")
part = ~full & (daily > 0)              # 다 운영하지 않았는데 합계는 0이 아닌 날
print("그런 날:", days[part], "/ 운영한 시간:", on[part].sum(axis=1))
print("하루 평균(365일):", daily.mean().round(), "→ (운영한 날만):", daily[full].mean().round())
print("어림 20,000 ÷ 실제:", round(20000 / daily[full].mean(), 2))

month = pd.to_datetime(days, format="%d/%m/%Y").month.to_numpy()   # 날마다 달
with0, only = [], []
for m in [6, 7, 8, 9, 10, 11]:
    pick = month == m
    with0.append(daily[pick].mean())
    only.append(daily[pick & full].mean())
    print(m, "월  0 포함:", round(with0[-1]), " 운영한 날만:", round(only[-1]), " 운영한 날:", (pick & full).sum())

x = np.arange(6)
plt.bar(x - 0.2, with0, width=0.4, label="0 포함", color="gray")
plt.bar(x + 0.2, only, width=0.4, label="운영한 날만", color="crimson")
plt.xticks(x, ["6월", "7월", "8월", "9월", "10월", "11월"])
plt.ylabel("하루 평균 대여량(대)")
plt.legend()
plt.show()

# ### 2-6
dow = (np.arange(365) + 4) % 7          # 0=월 … 6=일. 첫날(2017-12-01)은 금요일(4)
weekend = dow >= 5
wd = M[full & ~weekend].mean(axis=0)    # 운영한 평일의 시간대별 평균
we = M[full & weekend].mean(axis=0)     # 운영한 주말의 시간대별 평균
allday = M[full].mean(axis=0)           # 운영한 352일 모두
print("평일", (full & ~weekend).sum(), "일, 주말", (full & weekend).sum(), "일")
print("352일: 4시", allday[4].round(1), "/ 8시", allday[8].round(1), "/ 18시", allday[18].round(1))
print("8시  평일", wd[8].round(), " 주말", we[8].round())
print("15시 평일", wd[15].round(), " 주말", we[15].round())
print("가장 붐비는 시각 평일:", wd.argmax(), "시", wd.max().round(), "/ 주말:", we.argmax(), "시", we.max().round())
print("하루 합계 평일:", daily[full & ~weekend].mean().round(), "/ 주말:", daily[full & weekend].mean().round())

plt.plot(wd, marker="o", label="평일")
plt.plot(we, marker="s", label="주말")
plt.xticks(range(0, 24, 2))
plt.xlabel("시각(시)")
plt.ylabel("평균 대여량(대)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ### 2-7
temp = df["temp"].to_numpy()            # 기온 8,760개 (2-1의 temp는 숫자 하나였다)
temp_f = temp * 9 / 5 + 32              # 숫자 하나가 8,760칸에 퍼진다
i = temp.argmax()
print("가장 더운 시간:", df["date"][i], hour[i], "시", temp[i], "°C =", temp_f[i].round(2), "°F")
z = (temp - temp.mean()) / temp.std()   # 표준화: 평균을 빼고 표준편차로 나눈다
print("평균", temp.mean().round(2), "/ 표준편차(n으로 나눔)", temp.std().round(2),
      "/ (n − 1로 나눔)", temp.std(ddof=1).round(2))
print("첫 세 시간의 z:", z[:3].round(2), "/ z의 평균", abs(z.mean()).round(6), "/ 표준편차", z.std().round(6))
gap = M - allday                        # (365, 24) − (24,): 그 시각의 평소보다 얼마나 많았나
print(gap.shape, "/ 가장 많은 날 17~19시:", gap[d, 17:20].round())

plt.hist(z, bins=40)
plt.title("표준화한 기온 z")
plt.show()

# ### 2-8
import time

values = list(range(1_000_000))         # 0부터 999,999까지 리스트
arr = np.arange(1_000_000)              # 같은 숫자의 넘파이 배열

def loop_sum(xs):                       # 하나씩: 제곱해서 더하기
    total = 0
    for v in xs:
        total = total + v * v
    return total

def numpy_sum(a):                       # 한꺼번에
    return (a * a).sum()

def best_time(f, data, repeat=5):       # 다섯 번 재서 가장 빠른 시간(초)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        f(data)
        times.append(time.perf_counter() - start)
    return min(times)

print(loop_sum(values) == numpy_sum(arr))
t_loop, t_np = best_time(loop_sum, values), best_time(numpy_sum, arr)
print(f"리스트 반복 {t_loop * 1000:.1f}ms, 넘파이 {t_np * 1000:.2f}ms, 약 {t_loop / t_np:.0f}배")

# ### 2-9
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.images.shape)              # 이미지 1,797장, 한 장은 8×8
print(digits.images[0].astype(int))     # 첫 번째 그림 (정답: 0)
plt.imshow(digits.images[0], cmap="gray_r")
plt.show()

x = np.array([0.2, 0.9, 0.5])                        # 입력 3개
W = np.array([[0.5, -1.0, 2.0], [1.5, 0.3, -0.4]])   # 가중치 2×3
b = np.array([0.1, -0.2])                            # 편향 2개
print(W.shape, x.shape, (W @ x + b).round(2))

# ### M-1
x = [254, 204, 173, 107, 78]            # 2017년 12월 1일 0~4시 대여량
total = 0
for v in x:                             # Σ: i = 1부터 5까지 더하기
    total = total + v
print(total, total / len(x), np.mean(x))

# ### M-2
p = np.array([1500, 3200, 4800])
q = np.array([4, 2, 3])
print(p * q, p @ q, np.dot(p, q))

# ### M-3
A = np.array([[10, 6, 12, 4],
              [8, 5, 15, 2],
              [0, 0, 0, 0]])
print(A.shape, A.sum(axis=0), A.sum(axis=1))
print(A[:2].mean(axis=0))               # 셋째 날(0)을 빼고 시간대별 평균

# ### M-4
x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
dev = x - x.mean()                      # 평균과의 차이 (브로드캐스팅)
print(x.mean(), dev, (dev ** 2).sum())
print(np.sqrt((dev ** 2).sum() / len(x)), x.std(), x.std(ddof=1).round(2))
print("z:", ((x - x.mean()) / x.std()))

# ### MP-1
PATH = "kpx_시간별_전국_전력수요량_2021_2025.csv"   # 코랩 왼쪽 파일 창에 올린 파일
power = pd.read_csv(PATH)
print(power.shape)
print(power.iloc[:2, :6])
hours = [f"{h}시" for h in range(1, 25)]           # 열 이름 "1시" … "24시"
X = power[hours].to_numpy()                        # 날 × 24시간 행렬
print(X.shape, X.dtype)
bad = np.isnan(X).any(axis=1)                      # 빈칸이 하나라도 있는 행
print("빈칸이 있는 행:", bad.sum(), "/ 그 행의 날짜:", power["날짜"][bad].tolist())
X = X[~bad]                                        # 그 행을 빼고 다시 담기
dates = pd.to_datetime(power["날짜"][~bad]).reset_index(drop=True)
print(X.shape, dates.min().date(), "~", dates.max().date())

# ### MP-2
prof = X.mean(axis=0)                              # 1,826일의 시간대별 평균 (MW)
print("가장 많은 시각:", hours[prof.argmax()], prof.max().round(), "/ 가장 적은 시각:", hours[prof.argmin()], prof.min().round())
i, j = np.unravel_index(X.argmax(), X.shape)       # 가장 큰 칸의 (행, 열)
print("가장 큰 칸:", dates[i].date(), hours[j], X[i, j])
gwh = X.sum(axis=1) / 1000                         # 하루 전력량(GWh)
print("하루 평균:", gwh.mean().round(1), "GWh / 최대:", dates[gwh.argmax()].date(), gwh.max().round(1),
      "/ 최소:", dates[gwh.argmin()].date(), gwh.min().round(1))

# ### MP-3
m = dates.dt.month.to_numpy()
for name, pick in [("여름(7·8월)", (m == 7) | (m == 8)),
                   ("겨울(12·1·2월)", (m == 12) | (m == 1) | (m == 2)),
                   ("봄·가을(4·5·10월)", (m == 4) | (m == 5) | (m == 10))]:
    pr = X[pick].mean(axis=0)
    print(name, pick.sum(), "일 / 가장 많은 시각:", hours[pr.argmax()], pr.max().round(),
          "/ 가장 적은 시각:", hours[pr.argmin()], pr.min().round(), "/ 18시:", pr[17].round())
    plt.plot(range(1, 25), pr / 1000, marker="o", label=name)
plt.xticks(range(1, 25, 2))
plt.xlabel("시각(시)")
plt.ylabel("평균 전력 수요(GW)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ### MP-4
weekend_p = dates.dt.dayofweek.to_numpy() >= 5
print("하루 전력량 평일:", gwh[~weekend_p].mean().round(1), "GWh / 주말:", gwh[weekend_p].mean().round(1), "GWh")
print("주말은 평일의", round(gwh[weekend_p].mean() / gwh[~weekend_p].mean() * 100, 1), "%")
low = np.argsort(gwh)[:10]                         # 작은 값부터 열 개의 자리 번호
print("가장 적은 열흘:", dates[low].dt.strftime("%Y-%m-%d").tolist())

# ### EX-2
temps = [3.1, -2.5, 7.8, 5.0, -0.4]
best = temps[0]
for t in temps:
    if t > best:
        best = t
print(best)

# ### EX-7
T = temp.reshape(365, 24)
t_hour = T.mean(axis=0)                 # 365일의 시간대별 평균 기온
print(t_hour.round(1))
print("가장 따뜻한 시각:", t_hour.argmax(), "시", t_hour.max().round(1),
      "/ 가장 추운 시각:", t_hour.argmin(), "시", t_hour.min().round(1), "/ 18시:", t_hour[18].round(1))
