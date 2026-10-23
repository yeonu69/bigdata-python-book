# 01장. 우리는 이미 빅데이터 속에 산다 (개정판) — 05절 실습, 04절 확인 코드, 미니 프로젝트, 연습문제 정답 코드
# '# ### 1-k' 조각 하나가 책의 [파이썬 실습 1-k] 하나입니다.
# M-k는 04절 손 계산을 파이썬으로 확인하는 짧은 코드, MP-k는 [미니 프로젝트 1]의 단계, EX-k는 연습문제 정답 코드입니다.
# 실행: python3 practice_v2.py  (필요 라이브러리: pandas, matplotlib, koreanize-matplotlib)
# 실습 1-10과 미니 프로젝트는 강의 자료 파일(CSV)이 같은 폴더에 있어야 합니다.

# ### 1-1
import matplotlib.pyplot as plt
import koreanize_matplotlib     # 그래프의 한글이 깨지지 않게 (코랩에서 한 번 설치)

print("안녕하세요, 데이터!")
print(365 * 24)          # 1년은 몇 시간일까?
print(17000 * 365)       # 하루 17,000대씩 1년이면 몇 대일까?

week = [19082, 16696, 0, 22729, 23240, 6980, 10207]   # 2018년 4월 9일(월)~15일(일)
print("일주일 합:", sum(week), "/ 날수:", len(week))
plt.bar(["월", "화", "수", "목", "금", "토", "일"], week)
plt.title("2018년 4월 둘째 주의 하루 대여량")
plt.ylabel("대여량(대)")
plt.show()

# ### 1-2
import pandas as pd      # 표를 다루는 도구 판다스를 pd라는 별명으로 불러옵니다

URL = ("https://raw.githubusercontent.com/syedsharin/"
       "Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv")
df = pd.read_csv(URL, encoding="latin-1")     # 글자 저장 방식(인코딩)을 알려 줍니다
print(df.shape)                 # (행 수, 열 수)
print(df.columns.tolist())      # 열 이름 목록

df["Rented Bike Count"].head(168).plot()     # 앞의 168행 = 첫 7일
plt.title("첫 7일(168시간)의 시간별 대여량")
plt.xlabel("행 번호(시간)")
plt.show()

# ### 1-3
df.columns = ["date", "count", "hour", "temp", "humidity", "wind", "visibility",
              "dew", "solar", "rain", "snow", "season", "holiday", "functioning"]
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")   # 글자 → 날짜 (일/월/년)

print(df[["date", "hour", "count", "temp", "functioning"]].head(3))
print("날짜 수:", df["date"].nunique())
print("날짜마다 행 수:", df.groupby("date").size().unique())

daily = df.groupby("date")["count"].sum()     # 날짜마다 대여량을 더하기
print("하루 대여량이 0인 날:", (daily == 0).sum(), "일")
print(daily["2018-04-09":"2018-04-15"].tolist())   # [예제 1-1]의 한 주
daily.plot()
plt.title("하루 대여량 365일")
plt.ylabel("대여량(대)")
plt.show()

# ### 1-4
total = df["count"].sum()
print("1년 동안 빌린 횟수:", total)
print("어림셈 17,000 × 365:", 17000 * 365)
print("한 건을 100바이트로 잡은 원 기록 크기(MB):", round(total * 100 / 1_000_000))

season = df.groupby("season")["count"].sum()
print((season / total * 100).round(1))       # 계절마다 몇 %인가
(season / 10000).plot(kind="bar", rot=0)
plt.title("계절별 대여량 합계(만 건)")
plt.show()

# ### 1-5
print(df["functioning"].value_counts())
off = df[df["functioning"] == "No"]           # 운영하지 않은 시간만 고르기
print("운영하지 않은 날:", off["date"].nunique(), "일 / 그 시간의 대여량 합:", off["count"].sum())

on = df[df["functioning"] == "Yes"]           # 운영한 시간만 고르기
print("시간당 평균(모든 시간):", round(df["count"].mean(), 1))
print("시간당 평균(운영한 시간만):", round(on["count"].mean(), 1))
print("하루 평균(365일):", round(daily.mean()))
print("하루 평균(0인 날을 뺀", (daily > 0).sum(), "일):", round(daily[daily > 0].mean()))

plt.bar(["모든 시간", "운영한 시간만"], [df["count"].mean(), on["count"].mean()])
plt.title("0을 넣을 때와 뺄 때의 시간당 평균")
plt.ylabel("대여량(대)")
plt.show()

# ### 1-6
by_hour = on.groupby("hour")["count"].mean()     # 시각(0~23시)별 평균 대여량
plt.plot(by_hour.index, by_hour.values, marker="o")
plt.xticks(range(0, 24, 2))
plt.xlabel("시각(시)")
plt.ylabel("평균 대여량(대)")
plt.title("따릉이 시간대별 평균 대여량")
plt.grid(alpha=0.3)
plt.show()
print("가장 붐비는 시각:", by_hour.idxmax(), "시", round(by_hour.max()), "대")
print("가장 한가한 시각:", by_hour.idxmin(), "시", round(by_hour.min()), "대")

# ### 1-7
# 24시간을 모두 운영한 352일만 남기기 (2장과 같은 기준)
off_days = off["date"].unique()                       # 운영하지 않은 시간이 있는 13일
full = df[~df["date"].isin(off_days)].copy()
full["weekend"] = full["date"].dt.dayofweek >= 5      # 토요일(5)·일요일(6)이면 True
print("남은 날:", full["date"].nunique(), "/ 주말:", full[full["weekend"]]["date"].nunique())

weekday = full[~full["weekend"]].groupby("hour")["count"].mean()
weekend = full[full["weekend"]].groupby("hour")["count"].mean()
plt.plot(weekday, marker="o", label="평일")
plt.plot(weekend, marker="s", label="주말")
plt.xticks(range(0, 24, 2))
plt.xlabel("시각(시)")
plt.ylabel("평균 대여량(대)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print("8시 평일:", round(weekday[8]), "/ 주말:", round(weekend[8]))
print("가장 붐비는 시각 평일:", weekday.idxmax(), "시", round(weekday.max()),
      "/ 주말:", weekend.idxmax(), "시", round(weekend.max()))

# ### 1-8
import urllib.request
raw = urllib.request.urlopen(URL).read()      # 파일을 글자로 바꾸지 않고 바이트 그대로
print("파일 크기:", len(raw), "바이트")
print("41번째 바이트(0부터 세면 40번):", hex(raw[40]))
print(raw[25:45])
try:
    pd.read_csv(URL)                          # 아무 설정 없이 읽기
except UnicodeDecodeError as e:
    print("읽기 실패:", e)

# ### 1-9
vis = df["visibility"].value_counts()
print(vis.head(3))
print("가시거리 2000인 시간:", (df["visibility"] == 2000).sum(),
      "/", round((df["visibility"] == 2000).mean() * 100, 1), "%")
df["visibility"].plot(kind="hist", bins=40)
plt.title("가시거리(10m 단위)의 분포")
plt.show()

# ### 1-10
power = pd.read_csv("kpx_시간별_전국_전력수요량_2021_2025.csv")
subway = pd.read_csv("서울교통공사_서울전체_일별_시간대별_승하차_2023_2024.csv")
air = pd.read_csv("에어코리아_서울_도시대기평균_시간별_2023_2026.csv")
weather = pd.read_csv("asos_서울108_시간자료_2023_2026.csv")
for name, t in [("전력", power), ("지하철", subway), ("미세먼지", air), ("날씨", weather)]:
    print(name, t.shape)

hours = [f"{h}시" for h in range(1, 25)]
top = power[hours].max(axis=1).idxmax()               # 가장 많이 쓴 날의 행 번호
print("전력 수요 최대:", power.loc[top, hours].max(), "MW /", power.loc[top, "날짜"],
      power.loc[top, hours].idxmax())
ride = subway[subway["구분"] == "승차"]
print("지하철 하루 승차 2023년 평균:", round(ride[ride["날짜"] < "2024"]["합계"].mean()))
print("초미세먼지 평균(㎍/㎥):", round(air["PM25"].mean(), 1),
      "/ 나쁨 이상(36 이상) 시간의 비율:", round((air["PM25"] >= 36).mean() * 100, 1), "%")
print("기간:", weather["일시"].min(), "~", weather["일시"].max())
print("가장 더운 시각:", weather["기온(°C)"].max(), "°C / 가장 추운 시각:", weather["기온(°C)"].min(), "°C")

daily_gwh = power[hours].sum(axis=1) / 1000          # 하루 전력량(GWh)
plt.plot(pd.to_datetime(power["날짜"]), daily_gwh)
plt.title("전국 하루 전력 수요량(GWh)")
plt.show()

# ### M-1
week = [19082, 16696, 0, 22729, 23240, 6980, 10207]
print(sum(week), round(sum(week) / 7), round(sum(week) / 6))
print("차이:", round(sum(week) / 6) - round(sum(week) / 7))
print("6일 평균에 대해:", round((16489 - 14133) / 16489 * 100, 1), "%",
      "/ 7일 평균에 대해:", round((16489 - 14133) / 14133 * 100, 1), "%")

# ### M-2
weekday_share, weekend_share = 39.2, 27.8
print("%p:", round(weekday_share - weekend_share, 1))
print("%:", round((weekday_share - weekend_share) / weekend_share * 100))

# ### M-3
rush = [7, 8, 9, 17, 18, 19]
for name, part in [("평일", full[~full["weekend"]]), ("주말", full[full["weekend"]])]:
    share = part[part["hour"].isin(rush)]["count"].sum() / part["count"].sum() * 100
    print(name, "출퇴근 여섯 시간의 몫:", round(share, 1), "%")

# ### MP-1
PATH = "서울교통공사_서울전체_일별_시간대별_승하차_2023_2024.csv"   # 코랩 왼쪽 파일 창에 올린 파일
sub = pd.read_csv(PATH)
print(sub.shape)
print(sub.iloc[:2, :5])
print("기간:", sub["날짜"].min(), "~", sub["날짜"].max(), "/ 날짜 수:", sub["날짜"].nunique())
print(sub["구분"].value_counts())
print("빈칸:", sub.isna().sum().sum())

# ### MP-2
sub["날짜"] = pd.to_datetime(sub["날짜"])
ride = sub[sub["구분"] == "승차"].copy()
ride["달"] = ride["날짜"].dt.strftime("%Y-%m")
month = ride.groupby("달")["13-14시"].mean() / 10000          # 만 명 단위
print(month.round(1).to_string())
month.plot(marker="o")
plt.title("달마다 하루 평균 13~14시 승차(만 명)")
plt.xticks(rotation=90)
plt.show()

# ### MP-3
odd = ride["달"] == "2024-04"
print("2024년 4월 하루 합계 평균(만 명):", round(ride[odd]["합계"].mean() / 10000, 1))
print("다른 달 하루 합계 평균(만 명):", round(ride[~odd]["합계"].mean() / 10000, 1))
ok = ride[~odd].copy()                                       # 이상한 달을 뺀 표
ok["주말"] = ok["날짜"].dt.dayofweek >= 5
by_day = ok.groupby("주말")["합계"].mean()
print("하루 평균 승차: 평일", round(by_day[False]), "/ 주말", round(by_day[True]))
print("주말은 평일의", round(by_day[True] / by_day[False] * 100, 1), "%")
plt.bar(["평일", "주말"], by_day.values / 10000)
plt.title("하루 평균 승차(만 명, 2024년 4월 제외)")
plt.show()

# ### MP-4
print("2023년 1년 승차 합계:", ok[ok["날짜"].dt.year == 2023]["합계"].sum())
print("하루 승차 최대:", ok.loc[ok["합계"].idxmax(), "날짜"].date(), ok["합계"].max())
print("하루 승차 최소:", ok.loc[ok["합계"].idxmin(), "날짜"].date(), ok["합계"].min())
hours_sub = sub.columns[2:21]                                # 06시이전 ~ 23시이후
peak = ok[~ok["주말"]][hours_sub].mean()
print("평일에 가장 붐비는 시간대:", peak.idxmax(), round(peak.max()))

# ### EX-7
rainy = on["rain"] > 0
print("비 온 시간:", rainy.sum(), "/", len(on), "=", round(rainy.mean() * 100, 1), "%")
print(on.groupby(rainy)["count"].mean().round(1))
