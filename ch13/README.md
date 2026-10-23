# 13장 몸을 가진 AI: 센서 데이터로 읽는 피지컬 AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch13/13%EC%9E%A5_%ED%94%BC%EC%A7%80%EC%BB%ACAI_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`13장_피지컬AI_실습_colab_v2.ipynb`](13%EC%9E%A5_%ED%94%BC%EC%A7%80%EC%BB%ACAI_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch13.py`](practice_ch13.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 13-k` 조각 하나가 [파이썬 실습 13-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`asos_서울108_시간자료_2023_2026.csv`](data/asos_%EC%84%9C%EC%9A%B8108_%EC%8B%9C%EA%B0%84%EC%9E%90%EB%A3%8C_2023_2026.csv) | 종합 프로젝트 | 1.4MB | 기상청 기상자료개방포털, ASOS 서울(108) 시간자료 2023-01~2026-10을 7개 열로 정리 |  |
| [`kpx_시간별_전국_전력수요량_2021_2025.csv`](data/kpx_%EC%8B%9C%EA%B0%84%EB%B3%84_%EC%A0%84%EA%B5%AD_%EC%A0%84%EB%A0%A5%EC%88%98%EC%9A%94%EB%9F%89_2021_2025.csv) | 종합 프로젝트 A | 0.4MB | 한국전력거래소, 시간별 전국 전력수요량 2021~2025 (공공데이터포털)를 한 파일로 합침 |  |
| [`서울교통공사_서울전체_일별_시간대별_승하차_2023_2024.csv`](data/%EC%84%9C%EC%9A%B8%EA%B5%90%ED%86%B5%EA%B3%B5%EC%82%AC_%EC%84%9C%EC%9A%B8%EC%A0%84%EC%B2%B4_%EC%9D%BC%EB%B3%84_%EC%8B%9C%EA%B0%84%EB%8C%80%EB%B3%84_%EC%8A%B9%ED%95%98%EC%B0%A8_2023_2024.csv) | 종합 프로젝트 B | 0.2MB | 서울교통공사, 역별 일별 시간대별 승하차인원 2023~2024 (공공데이터포털)를 서울 전체 합계로 정리 |  |
| [`에어코리아_서울_도시대기평균_시간별_2023_2026.csv`](data/%EC%97%90%EC%96%B4%EC%BD%94%EB%A6%AC%EC%95%84_%EC%84%9C%EC%9A%B8_%EB%8F%84%EC%8B%9C%EB%8C%80%EA%B8%B0%ED%8F%89%EA%B7%A0_%EC%8B%9C%EA%B0%84%EB%B3%84_2023_2026.csv) | 종합 프로젝트 C | 1.1MB | 에어코리아 확정자료 2023-01~2026-04, 서울 25개 도시대기 측정소 시간별 평균 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| UCI_HAR_필요파일.zip | 본문 | 59.2MB | UCI 머신러닝 저장소 #240 Human Activity Recognition Using Smartphones에서 실습이 쓰는 파일(센서 신호 9채널, 정답, 참가자 번호, 특징 561개) | [바로 가기](https://raw.githubusercontent.com/bidhanbaray/UCI_HAR_Dataset/master/) |
| ai4i2020.csv | 13-6 | 0.5MB | UCI 머신러닝 저장소 #601 AI4I 2020 Predictive Maintenance | [바로 가기](https://raw.githubusercontent.com/lingchenlijing/AI4I-2020-Predictive-Maintenance-Dataset/master/data/ai4i2020.csv) |

교수용 정답 예시(종합 프로젝트 주제별 모범 풀이)는 이 저장소에 넣지 않았습니다.

## 파이썬 파일로 실행하기

```bash
cd ch13/data
python ../practice_ch13.py
```
