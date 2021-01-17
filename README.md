# transformer
트랜스포머를 직접 개발해보고 한글-영어 번역 태스크에 대해 실험. 
- 테스트 태스크: 한국-영어 번역
## Data
### AI Hub 한국어-영어 번역 샘플 데이터.
샘플 데이터의 경우 회원 가입 및 로그인 할 필요 없음.
- url: https://aihub.or.kr/sample_data_board

## Model

## Train
```
-----------------------------------------------------------------------------------------
| end of epoch   0 | time: 41902.65s | valid loss  0.70 | valid ppl     2.00
-----------------------------------------------------------------------------------------

```
## Issue
### 1. encoder 마스킹 에러
mask에 unsqueeze(1)을 통해 하나의 차원을 추가해줘야한다. 
```py
  mask = mask.unsqueeze(1)
  attention_score = attention_score.masked_fill(mask == 0, -1e9)
```


## References
- http://nlp.seas.harvard.edu/2018/04/03/attention