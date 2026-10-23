# 3장 지저분한 데이터 정리하기: 판다스와 데이터 정제

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch03/3%EC%9E%A5_%ED%8C%90%EB%8B%A4%EC%8A%A4%EC%99%80%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%A0%95%EC%A0%9C_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`3장_판다스와데이터정제_실습_colab_v2.ipynb`](3%EC%9E%A5_%ED%8C%90%EB%8B%A4%EC%8A%A4%EC%99%80%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%A0%95%EC%A0%9C_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch03.py`](practice_ch03.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 3-k` 조각 하나가 [파이썬 실습 3-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`bike_daily_clean.csv`](data/bike_daily_clean.csv) | 3-6에서 저장 | 31KB | 3장 실습이 만든 결과(일별) |  |
| [`bike_hourly_clean.csv`](data/bike_hourly_clean.csv) | 3-6에서 저장 | 0.7MB | 3장 실습이 따릉이 원본을 정제해 저장한 결과(시간별) |  |
| [`cleaning_log.csv`](data/cleaning_log.csv) | 3-8에서 저장 | 1KB | 3장 실습의 정제 기록표 |  |
| [`지하철_3개역_정제_2021_2024.csv`](data/%EC%A7%80%ED%95%98%EC%B2%A0_3%EA%B0%9C%EC%97%AD_%EC%A0%95%EC%A0%9C_2021_2024.csv) | 미니 프로젝트 정답 | 1.4MB | 미니 프로젝트 정제를 마친 정답 파일 |  |
| [`지하철_원자료_표본.zip`](data/%EC%A7%80%ED%95%98%EC%B2%A0_%EC%9B%90%EC%9E%90%EB%A3%8C_%ED%91%9C%EB%B3%B8.zip) | 미니 프로젝트 | 0.6MB | 서울교통공사 역별 일별 시간대별 승하차인원 원자료 6개 파일에서 세 역의 줄만 그대로 옮김(cp949) | zip 안에 CSV 6개 |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| SeoulBikeData.csv | 본문(정제 전 원본) | 0.6MB | UCI 머신러닝 저장소 #560 Seoul Bike Sharing Demand (서울 따릉이 2017-12~2018-11) | [바로 가기](https://raw.githubusercontent.com/syedsharin/Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch03/data
python ../practice_ch03.py
```
