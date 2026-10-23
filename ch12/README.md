# 12장 작년 이맘때처럼: 시계열 분석

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch12/12%EC%9E%A5_%EC%8B%9C%EA%B3%84%EC%97%B4_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`12장_시계열_실습_colab_v2.ipynb`](12%EC%9E%A5_%EC%8B%9C%EA%B3%84%EC%97%B4_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch12.py`](practice_ch12.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 12-k` 조각 하나가 [파이썬 실습 12-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`kpx_시간별_전국_전력수요량_2021_2025.csv`](data/kpx_%EC%8B%9C%EA%B0%84%EB%B3%84_%EC%A0%84%EA%B5%AD_%EC%A0%84%EB%A0%A5%EC%88%98%EC%9A%94%EB%9F%89_2021_2025.csv) | 미니 프로젝트 | 0.4MB | 한국전력거래소, 시간별 전국 전력수요량 2021~2025 (공공데이터포털)를 한 파일로 합침 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| airline-passengers.csv | 본문 | 2KB | 국제선 항공 승객 1949~1960 (Box & Jenkins) | [바로 가기](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) |
| opsd_germany_daily.csv | 본문 | 0.2MB | Open Power System Data, 독일 일별 전력 소비·풍력·태양광 2006~2017 | [바로 가기](https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch12/data
python ../practice_ch12.py
```
