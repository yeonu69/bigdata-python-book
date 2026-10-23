# 9장 스팸일까, 아닐까: 분류

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch09/9%EC%9E%A5_%EB%B6%84%EB%A5%98_%EC%8B%A4%EC%8A%B5_colab_v3.ipynb)

| 파일 | 설명 |
|---|---|
| [`9장_분류_실습_colab_v3.ipynb`](9%EC%9E%A5_%EB%B6%84%EB%A5%98_%EC%8B%A4%EC%8A%B5_colab_v3.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch09.py`](practice_ch09.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 9-k` 조각 하나가 [파이썬 실습 9-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`asos_서울108_시간자료_2023_2026.csv`](data/asos_%EC%84%9C%EC%9A%B8108_%EC%8B%9C%EA%B0%84%EC%9E%90%EB%A3%8C_2023_2026.csv) | 미니 프로젝트 | 1.4MB | 기상청 기상자료개방포털, ASOS 서울(108) 시간자료 2023-01~2026-10을 7개 열로 정리 |  |
| [`에어코리아_서울_도시대기평균_시간별_2023_2026.csv`](data/%EC%97%90%EC%96%B4%EC%BD%94%EB%A6%AC%EC%95%84_%EC%84%9C%EC%9A%B8_%EB%8F%84%EC%8B%9C%EB%8C%80%EA%B8%B0%ED%8F%89%EA%B7%A0_%EC%8B%9C%EA%B0%84%EB%B3%84_2023_2026.csv) | 미니 프로젝트 | 1.1MB | 에어코리아 확정자료 2023-01~2026-04, 서울 25개 도시대기 측정소 시간별 평균 |  |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| sms.tsv | 본문 | 0.5MB | UCI 머신러닝 저장소 #228 SMS Spam Collection | [바로 가기](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv) |

## 파이썬 파일로 실행하기

```bash
cd ch09/data
python ../practice_ch09.py
```
