import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 데이터 읽기
survey_data = pd.read_excel("유니폼_선호도_조사의 주소")

# 단일 응답 전처리
single_response_data = survey_data[['성별', '만나이', '야구를 좋아한 기간', '1년동안 직관 가는 횟수', '야구에 대한 애정도를 숫자로 표현해주세요.', '소유하고 있는 유니폼 개수 (숫자만 입력 부탁드립니다.)', '좋아하는 색깔', '본인의 퍼스널 컬러']].copy()

# 성별, 만나이 등의 범주형 데이터를 숫자로 변환
single_response_data['성별'] = single_response_data['성별'].map({'남자': 0, '여자': 1})
single_response_data['만나이'] = single_response_data['만나이'].map({'10대': 0, '20대': 1, '30대': 2, '40대': 3, '50대': 4})
single_response_data['야구를 좋아한 기간'] = single_response_data['야구를 좋아한 기간'].map({'1년 미만': 0, '3년 미만': 1, '6년 미만': 2, '10년 미만': 3, '10년 이상': 4, '태어날때 부터': 5})
single_response_data['1년동안 직관 가는 횟수'] = single_response_data['1년동안 직관 가는 횟수'].map({'10번 미만': 0, '10 ~ 20번': 1, '21 ~ 30번': 2, '31~ 40번': 3, '그 이상': 4})

single_response_data['좋아하는 색깔'] = single_response_data['좋아하는 색깔'].astype('category').cat.codes
single_response_data['본인의 퍼스널 컬러'] = single_response_data['본인의 퍼스널 컬러'].astype('category').cat.codes

# 소유하고 있는 유니폼 개수를 숫자로 변환
single_response_data['소유하고 있는 유니폼 개수 (숫자만 입력 부탁드립니다.)'] = single_response_data['소유하고 있는 유니폼 개수 (숫자만 입력 부탁드립니다.)'].fillna(0).astype(int)

single_response_data

# 복수 응답 전처리
def split_and_strip(column):
    return column.fillna('').str.split(',').apply(lambda x: [item.strip() for item in x if item.strip()])

# 좋아하는 유니폼 종류
favorite_uniforms = split_and_strip(survey_data['가장 좋아하는 유니폼 종류 3가지 선택'])
mlb_favorite_uniforms = MultiLabelBinarizer()
favorite_uniforms_encoded = mlb_favorite_uniforms.fit_transform(favorite_uniforms)

# 소유한 유니폼 종류
owned_uniforms = split_and_strip(survey_data['소유한 유니폼 중 2벌 이상 가지고 있는 유니폼 종류 (여러개면 모두 써주세요)'])
mlb_owned_uniforms = MultiLabelBinarizer()
owned_uniforms_encoded = mlb_owned_uniforms.fit_transform(owned_uniforms)

# 최종 데이터 결합
final_data = np.hstack((single_response_data.values, favorite_uniforms_encoded, owned_uniforms_encoded))

final_data

# 사용자 간의 유사도를 계산
similarity_matrix = cosine_similarity(final_data)

# 특정 사용자의 추천 리스트 생성 함수
def recommend_users(user_id, similarity_matrix, survey_data, top_n=3):
    user_similarity = similarity_matrix[user_id]
similar users
    similar_users = np.argsort(-user_similarity)

    # 유사도가 가장 높은 3명의 유저
    recommended_users = [u for u in similar_users if u != user_id][:top_n]

    favorite_uniforms = survey_data.iloc[recommended_users]['가장 좋아하는 유니폼 종류 3가지 선택']

    return list(zip(recommended_users, favorite_uniforms))

# 예시로 첫 번째 사용자의 추천 사용자 목록 출력
user_id = int(input("타겟 유저: "))
recommended_users = recommend_users(user_id, similarity_matrix, survey_data)
print(f"타겟 유저와 유사한 유저ID와 선택: {recommended_users}")