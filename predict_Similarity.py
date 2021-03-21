import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# sample data
text1 = """
게임 정말재밌음. 많은돈을 들이지않고도 6성을 얻을수도있고 대가리굴려서 몸비틀며 스테이지 깨는맛도있어서 좋음. 어느정도 성장하면 게임이 루즈해지긴하는데 중간중간 이벤트로 그걸 채워줌 그런데 최근 번역오류가 너무심함. 스토리적인 문맥오류나 오타정도는 그럴수있다고 생각함. 문젠 최근 협약때 수치를잘못적어 게임플레이에 지대한 영향을 미쳤고, 대족장이벤트때 훈장조건 표기오류, 7지역 하드 제약이 표기된것과 완전히 다른것 등 도를넘는 번역오류가 많이 발견됨. 페그오 사태때 명방에서 미리 번역오류를 발견했고 고치겠다 했을땐 일 열심히한다 생각했음. 근데 단순 번역오류를 이렇게 오래 수정하지 않는이유를 모르겠음. 위와같은 오류들은 당장 긴급패치로도 수정할수 있는것들인데 왜 미루는것인지 의문임
"""
data = pd.read_csv('./data/similar_data.csv', encoding='ANSI')
tfidf = TfidfVectorizer()

# 위의 내용을 바탕으로 코사인 유사도 함수식을 작성
def get_recommendations(review, cosine_sim=cosine_sim):
    # 입력한 리뷰로부터 해당되는 인덱스를 받아옴. 이제 선택한 리뷰를 가지고 연산
    idx = indices[review]

    # 모든 리뷰에 대해서 해당 리뷰와의 유사도를 구함
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 리뷰들을 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sorted(sim_scores, reverse=True)

    # 가장 유사한 10개의 리뷰를 받음
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 리뷰의 인덱스를 받음
    review_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 리뷰를 리턴
    return data.iloc[review_indices]

# 텍스트 지정해서 판별하기 
def similar_review(text):
    # 들어온 text를 data의 맨 마지막 행에 넣음
    data.loc[len(data)] = text

    # tfidf 수행
    tfidf_matrix = tfidf.fit_transform(data)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # 중복을 제거하면서 리뷰내용을 인덱스로, 기존 인덱스를 본문 데이터로 이동
    # 이는 리뷰를 입력하면 인덱스를 출력하기 위함
    indices = pd.Series(data1.index, index=data).drop_duplicates()
    indices.head()
    
    get_recommendations(text)

    return result, float(per)

if __name__ == '__main__':
    # debug code
    similar_review(text1)
