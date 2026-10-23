# 2장 숫자를 한꺼번에: 파이썬과 넘파이

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch02/2%EC%9E%A5_%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC%EB%84%98%ED%8C%8C%EC%9D%B4_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`2장_파이썬과넘파이_실습_colab_v2.ipynb`](2%EC%9E%A5_%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC%EB%84%98%ED%8C%8C%EC%9D%B4_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch02.py`](practice_ch02.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 2-k` 조각 하나가 [파이썬 실습 2-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`kpx_시간별_전국_전력수요량_2021_2025.csv`](data/kpx_%EC%8B%9C%EA%B0%84%EB%B3%84_%EC%A0%84%EA%B5%AD_%EC%A0%84%EB%A0%A5%EC%88%98%EC%9A%94%EB%9F%89_2021_2025.csv) | 미니 프로젝트 | 0.4MB | 한국전력거래소, 시간별 전국 전력수요량 2021~2025 (공공데이터포털)를 한 파일로 합침 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| SeoulBikeData.csv | 본문 | 0.6MB | UCI 머신러닝 저장소 #560 Seoul Bike Sharing Demand (서울 따릉이 2017-12~2018-11) | [바로 가기](https://raw.githubusercontent.com/syedsharin/Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch02/data
python ../practice_ch02.py
```
