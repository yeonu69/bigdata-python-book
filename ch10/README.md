# 10장 스무고개와 집단지성: 결정나무와 앙상블

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeonu69/bigdata-python-book/blob/main/ch10/10%EC%9E%A5_%ED%8A%B8%EB%A6%AC%EC%95%99%EC%83%81%EB%B8%94_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb)

| 파일 | 설명 |
|---|---|
| [`10장_트리앙상블_실습_colab_v2.ipynb`](10%EC%9E%A5_%ED%8A%B8%EB%A6%AC%EC%95%99%EC%83%81%EB%B8%94_%EC%8B%A4%EC%8A%B5_colab_v2.ipynb) | 실습 노트북. 책의 [파이썬 실습], 미니 프로젝트, 연습문제 정답 코드가 책과 같은 번호로 들어 있습니다. |
| [`practice_ch10.py`](practice_ch10.py) | 같은 코드를 파이썬 파일 하나로 모은 것. `# ### 10-k` 조각 하나가 [파이썬 실습 10-k] 하나입니다. |

## 인터넷에서 바로 읽는 데이터

아래 데이터는 이 저장소에 넣지 않았습니다. 노트북이 원래 주소에서 바로 읽습니다.

| 파일 | 쓰는 곳 | 크기 | 내용과 출처 | 주소 |
|---|---|---|---|---|
| bank-full.csv | 본문 | 4.6MB | UCI 머신러닝 저장소 #222 Bank Marketing (포르투갈 은행 전화 마케팅) | [바로 가기](https://raw.githubusercontent.com/khatrideepti/Bank-Marketing-Data-Analysis/master/bank-full.csv) |
| adult-all.csv | 미니 프로젝트 | 5.3MB | UCI 머신러닝 저장소 #2 Adult (1994 미국 인구조사 소득) | [바로 가기](https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv) |
