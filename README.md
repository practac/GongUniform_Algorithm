## 추천 알고리즘 사용법

### 1. 알고리즘 폴더 별 파일 다운로드
- 알고리즘 폴더 별로 `.ipynb` 파일과 `설문조사.xlsx` 혹은 `선수기록.xlsx` 파일을 다운로드합니다.
- `.py`는 구글 코랩이 어려울 경우 로컬에서 실행시키도록 한 파일입니다.



### 2. 파일 구글드라이브 업로드 및 주소 복사
- Google Drive에 각 설문조사, 선수기록 파일을 업로드합니다.
- **[설문기반]**`"/content/drive/MyDrive/해당설문조사파일명.xlsx"`에 있는 주소창에 본인의 드라이브에 업로드 된 *설문조사* 파일 주소를 붙입니다.
- **[기록기반]**`'/content/drive/MyDrive/Donuts/start/ml/batter_stats(1).csv'`에 있는 주소창에 본인의 드라이브에 업로드 된 *선수기록* 파일 주소를 붙입니다.
- "https://jimmy-ai.tistory.com/14" 참고



### 3. 구글 코랩 실행
- 구글 코랩에서 다음 알고리즘들을 실행합니다:
  - `설문기반_유니폼_추천.ipynb`
  - `설문기반_선수_추천.ipynb`
  - `기록기반_선수_추천.ipynb`


  
### 4. 설문기반 선수 추천 알고리즘 실행
- 원하는 유저 번호를 `타겟 유저` input 입력하면 유사한 유저들을 결과로 출력합니다.
- 유저 번호는 알고리즘 진행 중  `df = df[df['팀'] == '두산 베어스']` 이후의 `df` 결과표에 있는 row 숫자입니다. (0 ~ 18 가능)
- 프로토타입이 팀 '두산베어스'로 한정되어 유저도 두산베어스 팬으로 한정합니다.



### 5. 설문기반 유니폼 추천 알고리즘 실행
- 원하는 유저 번호를 `타겟 유저` input 입력하면 유사한 유저들을 결과로 출력합니다.
- 유저 번호는 `유니폼_선호도_조사.xlsx`의 `유저ID`를 참고하면 됩니다. ( 1 ~ 74 가능)



### 6. 기록기반 선수 추천 알고리즘 실행
