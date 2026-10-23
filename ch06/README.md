# 6장 파일에서, 웹에서: 데이터 불러오기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch06/6%EC%9E%A5_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%88%EB%9F%AC%EC%98%A4%EA%B8%B0_%EC%8B%A4%EC%8A%B5_colab.ipynb)

| 파일 | 설명 |
|---|---|
| [`6장_데이터불러오기_실습_colab.ipynb`](6%EC%9E%A5_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%88%EB%9F%AC%EC%98%A4%EA%B8%B0_%EC%8B%A4%EC%8A%B5_colab.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch06.py`](practice_ch06.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 6-k` 조각 하나가 [파이썬 실습 6-k] 하나입니다. |

## 이 저장소에 들어 있는 데이터 (`data/`)

노트북의 **데이터 받기** 셀이 이 파일들을 코랩으로 자동으로 내려받습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 읽을 때 |
|---|---|---|---|---|
| [`6장_강의자료.zip`](data/6%EC%9E%A5_%EA%B0%95%EC%9D%98%EC%9E%90%EB%A3%8C.zip) | 본문 전체, 미니 프로젝트 | 15.7MB | 에어코리아 최종확정 측정자료(2025년 서울 측정소, 월별 엑셀 12개), 기상청 기상자료개방포털 서울 시간자료 CSV(cp949), 한국전력거래소 시간별 전국 전력수요량 CSV 5개(2021~2025), 실습용 연습 웹사이트 | 노트북 둘째 셀이 압축을 풂 |

## 파이썬 파일로 실행하기

```bash
cd ch06/data
unzip -q -O utf-8 6장_강의자료.zip
python ../practice_ch06.py
```
