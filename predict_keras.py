#predict_keras.py

import pickle
import numpy as np
import keras
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

# 1점
text1 = """
게임 정말재밌음. 많은돈을 들이지않고도 6성을 얻을수도있고 대가리굴려서 몸비틀며 스테이지 깨는맛도있어서 좋음. 어느정도 성장하면 게임이 루즈해지긴하는데 중간중간 이벤트로 그걸 채워줌 그런데 최근 번역오류가 너무심함. 스토리적인 문맥오류나 오타정도는 그럴수있다고 생각함. 문젠 최근 협약때 수치를잘못적어 게임플레이에 지대한 영향을 미쳤고, 대족장이벤트때 훈장조건 표기오류, 7지역 하드 제약이 표기된것과 완전히 다른것 등 도를넘는 번역오류가 많이 발견됨. 페그오 사태때 명방에서 미리 번역오류를 발견했고 고치겠다 했을땐 일 열심히한다 생각했음. 근데 단순 번역오류를 이렇게 오래 수정하지 않는이유를 모르겠음. 위와같은 오류들은 당장 긴급패치로도 수정할수 있는것들인데 왜 미루는것인지 의문임
"""
# 5점
text2 = """
요즘에 몇몇 타겜들도 분위기가 저조하고 번역관련으로 민심이 많이 안좋아졌는데 하루빨리 고쳐져서 갈등이 해결됬으면 합니다... 비록 유저들의 비난이 쏟아지고 있지만 해결되면 수그러지는건 당연한 일이니까요 화이팅입니다'
"""

# 5점
text3 = """
'리듬게임 2년차 찍먹 해봤습니다. AP 한 번 해보고 왔는데 일단 명작 냄새 납니다. 판정선이 아르케아랑 비슷해서 금방 적응했습니다. 해금방식은 사볼마냥 뛰어야 하지만 감수할만하고 수록곡도 좋습니다. 세상에 크로스 소울이 있다니요! 나중에 컨플릭트나 프리덤 다이브같은 유명한 곡들도 수록되고 콜라보도 진행하면 엄청나겠는데요! 그리고 엄지모드와 다지모드가 따로 있습니다. 혁명적인데요? 리듬게임 입문용으로도 좋을것 같습니다.'
"""

# Keras 모델 정의하고 가중치 데이터 읽어 들이기 
loaded_model = load_model('./data/best_model.h5')
okt = Okt()
with open('./data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('./data/stopwords.pickle', 'rb') as f:
    stopwords = pickle.load(f) 
max_len = 80


# 텍스트 지정해서 판별하기 
def check_review(text):
    # !! 모델에서 max_len 바꿀시 꼭 바꿀것 !!
    # data = protext.process(text)

    data = okt.morphs(text) # 토큰화
    data = [word for word in data if not word in stopwords] # 불용어 제거

    encoded = tokenizer.texts_to_sequences([data]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩

    score = float(loaded_model.predict(pad_new)) # 예측
    if(score > 0.5):
        result = '긍정 리뷰'
        per = score * 100
    else:
        result = '부정 리뷰'
        per = (1 - score) * 100
    # print(result, score, per)
    return result, float(per)

if __name__ == '__main__':
    check_review(text1)
    check_review(text2)
    check_review(text3)