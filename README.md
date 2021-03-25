태양광 발전량 예측 AI 경진대회
=======================================

예시로 제시된 지역의 기상 데이터와 과거 발전량 데이터를 활용하여, 시간대별 태양광 발전량을 예측하는 대회입니다.  
대회 링크:
https://www.dacon.io/competitions/official/235680/overview/description/

Dataset
==================
이 저장소에 데이터셋은 제외되어 있습니다.  
데이터셋 출처: 
https://www.dacon.io/competitions/official/235680/data/

Structure
==================
```setup
.
└── main.py
└── evaluation.py
└── main.ipynb
└── weather-prediction.py
```
* main.py: feature extraction부터 modeling까지의 main문
* evaluation.py: validation 결과를 확인하고 ensemble하기 위한 파일
* main.ipynb: random forest를 이용한 baseline 파일  
    * 게시 링크: [link](https://www.dacon.io/competitions/official/235680/codeshare/2289?page=1&dtype=recent)
* weather-prediction.py: 날씨 예측을 위한 파일. error propagation으로 인해 활용하지는 않음

Results
==================
* 평가지표: Pinball Loss
* MSE 결과: 2.0
* private rank: 27/461