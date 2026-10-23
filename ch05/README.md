# 5장 우연인가 아닌가: 통계적 추론

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch05/5%EC%9E%A5_%ED%86%B5%EA%B3%84%EC%A0%81%EC%B6%94%EB%A1%A0_%EC%8B%A4%EC%8A%B5_colab_v3.ipynb)

| 파일 | 설명 |
|---|---|
| [`5장_통계적추론_실습_colab_v3.ipynb`](5%EC%9E%A5_%ED%86%B5%EA%B3%84%EC%A0%81%EC%B6%94%EB%A1%A0_%EC%8B%A4%EC%8A%B5_colab_v3.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch05.py`](practice_ch05.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 5-k` 조각 하나가 [파이썬 실습 5-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`asos_서울108_시간자료_2023_2026.csv`](data/asos_%EC%84%9C%EC%9A%B8108_%EC%8B%9C%EA%B0%84%EC%9E%90%EB%A3%8C_2023_2026.csv) | 미니 프로젝트 | 1.4MB | 기상청 기상자료개방포털, ASOS 서울(108) 시간자료 2023-01~2026-10을 7개 열로 정리 |  |
| [`서울교통공사_서울전체_일별_시간대별_승하차_2023_2024.csv`](data/%EC%84%9C%EC%9A%B8%EA%B5%90%ED%86%B5%EA%B3%B5%EC%82%AC_%EC%84%9C%EC%9A%B8%EC%A0%84%EC%B2%B4_%EC%9D%BC%EB%B3%84_%EC%8B%9C%EA%B0%84%EB%8C%80%EB%B3%84_%EC%8A%B9%ED%95%98%EC%B0%A8_2023_2024.csv) | 미니 프로젝트 | 0.2MB | 서울교통공사, 역별 일별 시간대별 승하차인원 2023~2024 (공공데이터포털)를 서울 전체 합계로 정리 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| SeoulBikeData.csv | 본문 | 0.6MB | UCI 머신러닝 저장소 #560 Seoul Bike Sharing Demand (서울 따릉이 2017-12~2018-11) | [바로 가기](https://raw.githubusercontent.com/syedsharin/Seoul-Bike-Sharing-Demand-Prediction/main/SeoulBikeData.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch05/data
python ../practice_ch05.py
```
