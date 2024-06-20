!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
# 한글 폰트 설치

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

batter_file_path = '선수기록파일주소.csv' # 타자 기록 data, STATIZ 사이트 웹 크롤링 http://www.statiz.co.kr/main.php
batter = pd.read_csv(batter_file_path, encoding='cp949') # 인코딩 오류 - encoding='cp949' 추가하여 해결

batter.columns

batter.head()

import matplotlib as mpl
mpl.rc('font', family='NanumBarunGothic') # 한글 폰트(그래프)

batter['비율 wRC+(2023)'].describe()

batter['비율 wRC+(2023)'].hist(bins=100) # 히스토그램 분포

batter_wRC_df = batter[['G', '타석', '타수', '득점', '안타', '2타', '3타', '홈런', '루타',
       '도루', '도실', '볼넷', '사구', '고4', '삼진', '병살', '비율 타율', '비율 출루',
       '비율 장타', '비율 OPS', '비율 wOBA', '비율 wRC+(2022 전)', '비율 wRC+(2023)', 'WAR*', 'WPA']]

def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20, 16]
    fig = plt.figure(1)

    for i in range(len(df.columns)): # df의 column 갯수 만큼의 subplot을 출력
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i])
    plt.show()

plot_hist_each_column(batter_wRC_df) # 히스토그램 분포에서 비율 wRC+와 분포가 유사한 것은 비율 출루, 비율 장타, 비율 OPS, 비율 wOBA, 안타라는 것 확인

pd.options.mode.chained_assignment = None # float 형태로

def standard_scaling(df, scale_columns): # scailing 함수
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
    return df

scale_columns = ['G', '타석', '타수', '득점', '안타', '2타', '3타', '홈런', '루타',
       '도루', '도실', '볼넷', '사구', '고4', '삼진', '병살', '비율 타율', '비율 출루',
       '비율 장타', '비율 OPS', '비율 wOBA', '비율 wRC+(2022 전)', 'WAR*', 'WPA']
batter_df = standard_scaling(batter, scale_columns) # 각 column에 대하여 scaling 수행

batter_df = batter_df.rename(columns={'비율 wRC+(2024)': 'y'}) # 예측하려는 부분('비율 wRC+(2023)' column) 'y'로
batter_df.head(5)

team_encoding = pd.get_dummies(batter_df['팀']) # 10개 팀을 one-hot encoding으로 변환
batter_df = batter_df.drop('팀', axis=1)
batter_df = batter_df.join(team_encoding)

team_encoding.head(5)

batter_df.head(5)

batter_df = batter_df.dropna(axis=0) # 결측값이 0인 행 삭제
print(batter_df.shape)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

X = batter_df[batter_df.columns.difference(['이름', 'y'])]
y = batter_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19) # train data : test data = 8 : 2

lr = linear_model.LinearRegression() # 회귀 모델 학습
model = lr.fit(X_train, y_train)

print(lr.coef_) # 회귀 분석 계수 출력

!pip install statsmodels
import statsmodels.api as sm
# statsmodel 라이브러리로 회귀 분석

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
model.summary()

plt.rcParams['figure.figsize'] = [20, 16]

# 회귀 분석 계수를 리스트 형태로
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)

# 변수명을 리스트 형태로
x_labels = model.params.index.tolist()

# 회귀 계수 출력
ax = coefs_series.plot(kind='bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)

X = batter_df[batter_df.columns.difference(['이름', 'y'])]
y = batter_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# 회귀 분석 모델을 학습
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 회귀 분석 모델을 평가
print(model.score(X_train, y_train)) # train R2 score
print(model.score(X_test, y_test)) # test R2 score

# 가장 큰 영향을 미치는 feature 선정
X = batter_df[['비율 OPS', '비율 wOBA', '비율 장타', '비율 출루', '안타']]
y = batter_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# 모델 학습
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 모델 평가
print(model.score(X_train, y_train)) # train R2 score
print(model.score(X_test, y_test)) # test R2 score

# 2023년 wRC+를 예측하여 '예측 2023 wRC+' column 생성
X = batter_df[['비율 OPS', '비율 wOBA', '비율 장타', '비율 출루', '안타']]
batter_2023_wRC = lr.predict(X)
batter_df['예측 2023 wRC+'] = pd.Series(batter_2023_wRC)

# 원래의 데이터 프레임 다시 로드
batter_file_path = '../content/drive/MyDrive/Donuts/start/ml/batter_stats.csv'
batter = batter[['이름']]
result_df = batter_df.sort_values(by=['y'], ascending=False)
result_df = result_df.merge(batter, on=['이름'], how='left')
result_df = result_df[['이름', 'y', '예측 2023 wRC+']]
result_df.columns = ['선수명', '실제 2023 wRC+', '예측 2023 wRC+']

result_df = result_df.dropna(axis=0) # 결측값이 0인 행 삭제
print(result_df)

# 선수별 wRC+ 정보(실제 2023 wRC+, 예측 2023 wRC+)를 bar 그래프로 출력
result_df.plot(x='선수명', y=['실제 2023 wRC+', '예측 2023 wRC+'], kind="bar")
