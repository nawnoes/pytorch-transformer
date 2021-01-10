# transformers
트랜스포머 직접 작성해보기.
- 테스트 태스크: 한국-영어 번역
## 데이터
### AI Hub 한국어-영어 번역 샘플 데이터.
샘플 데이터의 경우 다운로드 할 필요 없음.
- url: https://aihub.or.kr/sample_data_board



## 버그
### 1. encoder 마스킹 에러
mask에 unsqueeze(1)을 통해 하나의 차원을 추가해줘야한다. 
```py
  mask = mask.unsqueeze(1)
  attention_score = attention_score.masked_fill(mask == 0, -1e9)
```

