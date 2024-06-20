import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("선수_선호_조사의 파일 주소")

df

# 결측치 확인하고 결측치 수 순서대로 정렬
#missing_values = df.isnull().sum()
#missing_values.sort_values(ascending=False, inplace=True)

# 결측치 있는 열 확인
#missing_values[missing_values > 0]

df.drop('타임스탬프', axis = 1, inplace = True)

df.drop('애정도', axis = 1, inplace = True)
df.drop('성별', axis = 1, inplace = True)
df.drop('만나이', axis = 1, inplace = True)
df.drop('좋아하는 선수', axis = 1, inplace = True)
df.drop('포지션', axis = 1, inplace = True)
df.drop('기간', axis = 1, inplace = True)
df.head()

df.drop("4. 가장 중요하게 생각하는 요소를 '3개만' 선택해주세요", axis = 1, inplace = True)
df.drop('1. 선수 추천 알고리즘에 대해 추가로 건의하실 사항이 있으시면 적어주세요 :)', axis = 1, inplace = True)
df.drop('2. 기프티콘을 수령하실 연락처를 적어주세요.', axis = 1, inplace = True)
df.head()

df.fillna(0, inplace=True)
df.head()

df = df[df['팀'] == '두산 베어스']

# 인덱스 열 추가
df.index.name = '전체조사번호'
df.reset_index(inplace=True)

df

player_row = int(input("타겟 유저: "))
df.iloc[player_row]

df.drop('팀', axis = 1, inplace = True)
df

target_user = df.iloc[player_row, 2:].values.reshape(1, -1)
cosine_sim = cosine_similarity(df.iloc[:, 2:], target_user)
df['유사도'] = cosine_sim

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 50)
df

df = df[df['선수 이름'] != 0] # 선수 이름 0 인 값 제거
sorted_df = df.sort_values(by='유사도', ascending=False)
top_similar_players = sorted_df.head(10)

if top_similar_players.duplicated(subset=['선수 이름']).any():
    unique_players_df = top_similar_players.drop_duplicates(subset=['선수 이름'], keep='first')
    result = unique_players_df
else:
    result = top_similar_players
result

player_names = result['선수 이름']
print(player_names.iloc[0] + "과 비슷한 선수 추천 순위")

for i, player_name in enumerate(player_names.iloc[1:], start=1):
    print(f"{i} 순위: {player_name}")