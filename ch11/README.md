# 11장 리뷰 속 마음 읽기: 텍스트 분석

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch11/11%EC%9E%A5_%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%B6%84%EC%84%9D_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`11장_텍스트분석_실습_colab_v2.ipynb`](11%EC%9E%A5_%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%B6%84%EC%84%9D_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch11.py`](practice_ch11.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 11-k` 조각 하나가 [파이썬 실습 11-k] 하나입니다. |
| [텍스트 분석 시뮬레이터](https://yeonu69.github.io/bigdata-python-book/sim/ch11/) | 책 04절의 [시뮬레이터로 해 보기]. 웹 브라우저에서 TF-IDF 계산, 코사인 유사도, 로지스틱 회귀의 조각 점수 합, 확신과 망설임(사람에게 넘길 구간)을 직접 움직여 봅니다. 책의 영화 리뷰 모델(시험 정확도 87.1%)로 계산한 값이 들어 있습니다. 파일: [`sim/ch11/index.html`](../sim/ch11/index.html) |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| ratings_train.txt | 본문 | 14.6MB | 네이버 영화 리뷰 말뭉치 NSMC v1.0 학습용 (CC0) | [바로 가기](https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt) |
| ratings_test.txt | 본문 | 4.9MB | NSMC 시험용 (CC0) | [바로 가기](https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt) |
| federalist.rda | 11-11 | 0.3MB | 『페더럴리스트 페이퍼』 85편 (R corpus 패키지) | [바로 가기](https://raw.githubusercontent.com/cran/corpus/master/data/federalist.rda) |
| naver_shopping.txt | 미니 프로젝트 | 20.6MB | 네이버 쇼핑 리뷰 20만 개 (bab2min/corpus) | [바로 가기](https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt) |
