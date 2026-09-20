# ANN 기반 기대수명 예측

- 목표: 국가별 보건·사회경제 지표를 이용한 기대수명 예측 및 은닉 노드 수에 따른 성능 비교
- Target: Life expectancy, 년 단위
- 모델: 단일 은닉층 ANN, 선형회귀
- 평가 지표: MAE, RMSE, R², Pearson 상관계수

## 파일 구성

```text
life-expectancy-ann/
├─ README.md
├─ requirements.txt
├─ notebooks/
│  └─ life_expectancy_ann.ipynb
└─ src/
   ├─ __init__.py
   ├─ download_data.py
   ├─ anova.py
   ├─ model.py
   └─ train.py
```

- [실험 노트북](notebooks/life_expectancy_ann.ipynb): 데이터 확인, ANOVA, 모델 비교, 평가 결과
- [download_data.py](src/download_data.py): 데이터 다운로드
- [anova.py](src/anova.py): 입력 후보 18개의 순차 분산분석
- [model.py](src/model.py): ANN 구조
- [train.py](src/train.py): 전처리, 학습, 검증 기반 모델 선택, 평가 및 시각화

## 데이터

- 출처: [Kaggle Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- 원본: 2000–2015년 국가별 데이터, 2,938행·22열
- ANN 입력: 보건·사회경제 지표 14개
- ANOVA: 후보 18개와 타깃의 결측 행 제거 후 1,649행 사용
- ANN: 입력 14개와 타깃의 결측 행 제거 후 1,855행 사용
- 분할: 학습 1,113행·검증 371행·테스트 371행, seed=42
- 정규화: 학습 데이터 기준 입력 및 타깃의 Min–Max 정규화
- 변수 구성: ANOVA 표를 통한 선정 근거 확인, ANN 입력 14개 고정
- 평가 범위: 무작위 행 분할, 동일 국가의 다른 연도 포함 가능
- 원본 CSV의 저장소 미포함, 실행 전 다운로드 필요

## 실행 환경

- 검증 환경: Windows, Python 3.11.7, CPU
- 노트북 실행: VS Code의 Python 및 Jupyter 확장 또는 JupyterLab

| 라이브러리 | 검증 버전 | 용도 |
|---|---|---|
| numpy | 1.26.4 | 수치 연산 |
| pandas | 2.2.3 | 데이터 처리 |
| scikit-learn | 1.5.2 | 전처리, 선형회귀, 평가 |
| statsmodels | 0.14.6 | ANOVA |
| matplotlib | 3.9.2 | 시각화 |
| torch | 2.5.1+cpu | ANN 구성 및 학습 |
| ipykernel | 6.29.5 | Python 노트북 커널 |
| jupyterlab | 4.3.4 | 노트북 실행 환경 |

- 전체 직접 의존성과 버전: [requirements.txt](requirements.txt)
- nbformat, nbclient: 노트북 처리 및 실행 검증 도구

## 실행 방법

### 1. 가상환경 생성 및 라이브러리 설치

- 저장소 루트에서 실행

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- macOS/Linux의 Python 실행 경로: `.venv/bin/python`

### 2. 데이터 다운로드

```powershell
.venv\Scripts\python.exe -m src.download_data
```

- 저장 위치: `data/Life Expectancy Data.csv`
- 다운로드 실패 시 Kaggle 출처 페이지에서 CSV를 받아 동일 경로에 저장

### 3. 노트북 실행

1. VS Code에서 저장소 폴더 열기
2. `notebooks/life_expectancy_ann.ipynb` 열기
3. 커널 선택에서 `.venv`의 Python 선택
4. 커널 재시작 후 전체 셀 실행

JupyterLab 사용 시 커널 등록 및 실행:

```powershell
.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name life-expectancy-ann --display-name "Python (life-expectancy-ann)"
.venv\Scripts\python.exe -m jupyterlab
```

- JupyterLab 커널: `Python (life-expectancy-ann)`

### 4. 명령줄 실행

```powershell
.venv\Scripts\python.exe -m src.anova
.venv\Scripts\python.exe -m src.train
```

- ANN 비교: 은닉 노드 5·10·15개, 검증 RMSE 기준 선택
- 결과 저장 위치: 실행 중 자동 생성되는 `results/`
- 노트북에 저장된 표와 그래프를 통한 실행 결과 확인
- 노트북 재실행 시 결과 파일과 출력 갱신
- 명령줄 실행 시 결과 파일 갱신, 노트북 출력은 별도 실행 필요
