# 7장 끼리끼리 모으기: 군집화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch07/7%EC%9E%A5_%EA%B5%B0%EC%A7%91%ED%99%94_%EC%8B%A4%EC%8A%B5_colab_v3.ipynb)

| 파일 | 설명 |
|---|---|
| [`7장_군집화_실습_colab_v3.ipynb`](7%EC%9E%A5_%EA%B5%B0%EC%A7%91%ED%99%94_%EC%8B%A4%EC%8A%B5_colab_v3.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch07.py`](practice_ch07.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 7-k` 조각 하나가 [파이썬 실습 7-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`서울지하철_역별_시간대별_월평균_승하차_2024.csv`](data/%EC%84%9C%EC%9A%B8%EC%A7%80%ED%95%98%EC%B2%A0_%EC%97%AD%EB%B3%84_%EC%8B%9C%EA%B0%84%EB%8C%80%EB%B3%84_%EC%9B%94%ED%8F%89%EA%B7%A0_%EC%8A%B9%ED%95%98%EC%B0%A8_2024.csv) | 미니 프로젝트 | 1.9MB | 서울교통공사 역별 일별 시간대별 승하차인원 2024를 역·월·날종류별 평균으로 정리 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| Significant_Earthquakes.csv | 본문 | 21.1MB | USGS 기반 Kaggle Significant Earthquakes의 깃허브 사본 (마지막 사건 2026-09-27) | [바로 가기](https://raw.githubusercontent.com/Abdullah-Moataz/Earthquakes/cf0d033d35f5d45eef6175bab6c60829363d6de4/Significant_Earthquakes.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch07/data
python ../practice_ch07.py
```
