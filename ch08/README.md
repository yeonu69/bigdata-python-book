# 8장 내일의 발전량: 회귀 분석

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch08/8%EC%9E%A5_%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`8장_회귀분석_실습_colab_v2.ipynb`](8%EC%9E%A5_%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch08.py`](practice_ch08.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 8-k` 조각 하나가 [파이썬 실습 8-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`asos_서울108_시간자료_전체열_2023_2026.csv`](data/asos_%EC%84%9C%EC%9A%B8108_%EC%8B%9C%EA%B0%84%EC%9E%90%EB%A3%8C_%EC%A0%84%EC%B2%B4%EC%97%B4_2023_2026.csv) | 본문, 미니 프로젝트 | 4.4MB | 기상청 기상자료개방포털, ASOS 서울(108) 시간자료 2023-01~2026-10, 원자료 38개 열 그대로 |  |
| [`kpx_시간별_전국_전력수요량_2021_2025.csv`](data/kpx_%EC%8B%9C%EA%B0%84%EB%B3%84_%EC%A0%84%EA%B5%AD_%EC%A0%84%EB%A0%A5%EC%88%98%EC%9A%94%EB%9F%89_2021_2025.csv) | 미니 프로젝트 | 0.4MB | 한국전력거래소, 시간별 전국 전력수요량 2021~2025 (공공데이터포털)를 한 파일로 합침 |  |
| [`kpx_지역별_시간별_태양광풍력_발전량_2025.csv`](data/kpx_%EC%A7%80%EC%97%AD%EB%B3%84_%EC%8B%9C%EA%B0%84%EB%B3%84_%ED%83%9C%EC%96%91%EA%B4%91%ED%92%8D%EB%A0%A5_%EB%B0%9C%EC%A0%84%EB%9F%89_2025.csv) | 본문 | 7.1MB | 한국전력거래소, 지역별 시간별 태양광·풍력 발전량 2025 (공공데이터포털) |  |
| [`출처.txt`](data/%EC%B6%9C%EC%B2%98.txt) | - | 2KB | 8장 데이터의 출처와 정리 방법 설명 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| GaltonFamilies.csv | 8-1 개념 | 35KB | 골턴의 부모·자녀 키 기록 (R HistData 패키지) | [바로 가기](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/HistData/GaltonFamilies.csv) |

## 파이썬 파일로 실행하기

```bash
cd ch08/data
python ../practice_ch08.py
```
