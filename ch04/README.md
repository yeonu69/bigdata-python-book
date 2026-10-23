# 4장 그림으로 데이터와 대화하기: 탐색적 분석과 시각화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch04/4%EC%9E%A5_%ED%83%90%EC%83%89%EC%A0%81%EB%B6%84%EC%84%9D%EA%B3%BC%EC%8B%9C%EA%B0%81%ED%99%94_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`4장_탐색적분석과시각화_실습_colab_v2.ipynb`](4%EC%9E%A5_%ED%83%90%EC%83%89%EC%A0%81%EB%B6%84%EC%84%9D%EA%B3%BC%EC%8B%9C%EA%B0%81%ED%99%94_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch04.py`](practice_ch04.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 4-k` 조각 하나가 [파이썬 실습 4-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`서울_미세먼지_날씨_시간별_2023_2025.csv`](data/%EC%84%9C%EC%9A%B8_%EB%AF%B8%EC%84%B8%EB%A8%BC%EC%A7%80_%EB%82%A0%EC%94%A8_%EC%8B%9C%EA%B0%84%EB%B3%84_2023_2025.csv) | 미니 프로젝트 | 1.5MB | 에어코리아 서울 25개 측정소 평균 PM10·PM2.5 + 기상청 ASOS 서울(108) 날씨를 시각으로 합침 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| SeoulBikeData.csv | 본문 | 0.6MB | UCI 머신러닝 저장소 #560 Seoul Bike Sharing Demand (서울 따릉이 2017-12~2018-11) | [바로 가기](https://raw.githubusercontent.com/syedsharin/Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch04/data
python ../practice_ch04.py
```
