import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# sample data
text1 = """
게임 종료시 자잘한 소리가 지속적으로 나서 항상 앱 종료할 때마다 앱 자체를 완전히 종료해야 한다는 점을 제외한다면 전체적으로 만족스러움. 노트20ultra인데 렉이 약간 발생하나, 저장공간 관리를 하지 않은 내 탓 일 수도 있다 봄.
"""

def similar_review(text):
    # 데이터를 불러오면서 리뷰 열만 가져오기
    data = pd.read_csv('./data/similar_data.csv', encoding='ANSI')
    
    # input받은 text를 review 변수로 재지정
    review = text

    # review를 data에 넣기
    data.loc[len(data)] = review

    # 중복값 처리 및 index 초기화
    data.drop_duplicates('REVIEW', keep="first", inplace=True)
    data.reset_index(inplace=True)

    # review 열만 가져와서 series로 만들기
    data_review = data['REVIEW']

    tfidf = TfidfVectorizer()
    # 리뷰 데이터에 대해서 tf-idf 수행
    tfidf_matrix = tfidf.fit_transform(data_review)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # 중복을 제거하면서 리뷰내용을 인덱스로, 기존 인덱스를 본문 데이터로 이동
    # 이는 리뷰를 입력하면 인덱스를 출력하기 위함
    indices = pd.Series(data_review.index, index=data_review).drop_duplicates()
    indices.head()
    # 위의 내용을 바탕으로 코사인 유사도 함수식을 작성
    def get_recommendations(review, cosine_sim=cosine_sim):
        # 입력한 리뷰로부터 해당되는 인덱스를 받아옴. 이제 선택한 리뷰를 가지고 연산
        idx = indices[review]

        # 모든 리뷰에 대해서 해당 리뷰와의 유사도를 구함
        sim_scores = list(enumerate(cosine_sim[idx]))

        # 유사도에 따라 리뷰들을 정렬
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # sim_scores = sorted(sim_scores, reverse=True)

        # 가장 유사한 10개의 리뷰를 받음
        sim_scores = sim_scores[1:11]

        # 가장 유사한 10개의 리뷰의 인덱스를 받음
        review_indices = [i[0] for i in sim_scores]

        # 가장 유사한 10개의 리뷰를 리턴
        return data_review.iloc[review_indices]

    get_recommendations(review)
    result = get_recommendations(review)

    result = result.to_list()
    return result

if __name__ == '__main__':
    # module test code
    final_result = similar_review(text1)
    print(final_result)
