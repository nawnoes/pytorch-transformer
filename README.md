# Transformer
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention)와 [pytorch 공식문서](https://tutorials.pytorch.kr/beginner/transformer_tutorial.html)를 
참고하여 트랜스포머를 직접 개발해보고 한글-영어 번역 태스크에 대해 학습 및 테스트.
![](./images/transformer-translation.png)
## Data
### ① AI Hub 한국어-영어 번역 샘플 데이터.
> 샘플 데이터의 경우 회원 가입 및 로그인 할 필요 없음.
- url: https://aihub.or.kr/sample_data_board
### ② AI Hub 한국어-영어 번역 데이터 - 구어체
사용허가를 받아 다운로드 후 구어체 데이터데 대하여 학 
## Model

```text
model
 ㄴ transformer.py    ﹒﹒﹒ 트랜스포머 모델
 ㄴ util.py           ﹒﹒﹒ 모델에 사용되는 유틸
 ㄴ visualization.py  ﹒﹒﹒ 모델의 시각화에 사용하는 습
train_translation.py﹒﹒﹒ 한국어-영어 번역 학습
run_translation.py  ﹒﹒﹒ 한국어-영어 번역 테스
```

### Train Setting
```python
# Model setting
model_name = 'transformer-translation-spoken'
vocab_num = 22000
max_length = 64
d_model = 512
head_num = 8
dropout = 0.1
N = 6
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Hyperparameter
epochs = 50
batch_size = 8
learning_rate = 0.8
```

### Train Result
- Epoch: 50

```
-----------------------------------------------------------------------------------------
| end of epoch   0 | time: 2005.31s | valid loss  4.95 | valid ppl   141.70
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 2149.59s | valid loss  4.62 | valid ppl   101.26
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 2058.49s | valid loss  4.39 | valid ppl    80.86
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 1966.75s | valid loss  4.25 | valid ppl    70.38
-----------------------------------------------------------------------------------------
                                ...중략...
| end of epoch  47 | time: 1973.69s | valid loss  2.79 | valid ppl    16.26
-----------------------------------------------------------------------------------------
| end of epoch  48 | time: 2076.40s | valid loss  2.77 | valid ppl    16.00
-----------------------------------------------------------------------------------------
| end of epoch  49 | time: 2080.24s | valid loss  2.79 | valid ppl    16.26
-----------------------------------------------------------------------------------------
```


## Issue
### 1. encoder 마스킹 에러
mask에 unsqueeze(1)을 통해 하나의 차원을 추가해줘야한다. 
```py
  mask = mask.unsqueeze(1)
  attention_score = attention_score.masked_fill(mask == 0, -1e9)
```
### 2. max_seq_len이 긴경우
대부분의 문장이 짧게 구성되어 있다. 최초에 512 토근으로 지정후 학습하면, 시간도 느리고, pad 토큰에 대해 학습하여
성능이 좋지 않았다.


## References
- http://nlp.seas.harvard.edu/2018/04/03/attention