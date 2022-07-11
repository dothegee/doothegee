#서포트벡터머신(Support Vector Machines)
# 회귀, 분류, 이상치 탐지 등에 사용되는 지도 학습 방법
# 클래스 사이의 경계에 위치한 데이터 포인트를 서포트 벡터(Support Vector)라고 함
# 각서포트 벡터가 클래스 사이의 결정 경계를 구분하는데 얼마나 중요한지를 학습
# 각 서포트 벡터 사이의 마진이 가장 큰 방향으로 학습
# 서포트 벡터 까지의 거리와 서포트 벡터의 중요도를 기반으로 예측을 수행


# import mulitprocessing
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
plt.style.use(['seaborn-whitegrid'])

from sklearn.svm import SVR, SVC
from sklearn.datasets import load_boston, load_diabetes
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

#####SVM을 사용한 회귀 모델(SVR)
X,y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
model = SVR()
model.fit(X_train,y_train)

print(f'학습 데이터 점수 : {model.score(X_train, y_train)}')

print(f'평가 데이터 점수 : {model.score(X_test, y_test)}')



#####SVM을 사용한 분류 모델 (SVC)
X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
model = SVC()
model.fit(X_train,y_train)

print(f'학습 데이터 점수 : {model.score(X_train, y_train)}')

print(f'평가 데이터 점수 : {model.score(X_test, y_test)}')

#####커널 기법
#입력 데이터를 고차원 공간에 사상해서 비선형 특징을 학습할 수 있도록 확장하는 방법
#scikit-learn에서 Linear,Polynomial, RBF(Radial Basis Function) 지원

#SVR

X,y = load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)

print(f'linear_svr학습 데이터 점수 : {linear_svr.score(X_train, y_train)}')
print(f'linear_svr평가 데이터 점수 : {linear_svr.score(X_test, y_test)}')

polynomial_svr = SVR(kernel='poly')
polynomial_svr.fit(X_train,y_train)

print(f'polynomial_svr학습 데이터 점수 : {polynomial_svr.score(X_train, y_train)}')
print(f'polynomial_svr평가 데이터 점수 : {polynomial_svr.score(X_test, y_test)}')

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)

print(f'rbf_svr학습 데이터 점수 : {rbf_svr.score(X_train, y_train)}')
print(f'rbf_svr평가 데이터 점수 : {rbf_svr.score(X_test, y_test)}')


#SVC

X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

linear_svc = SVC(kernel='linear')
linear_svc.fit(X_train,y_train)

print(f'linear_svc학습 데이터 점수 : {linear_svc.score(X_train, y_train)}')
print(f'linear_svc평가 데이터 점수 : {linear_svc.score(X_test, y_test)}')

polynomial_svc = SVC(kernel='poly')
polynomial_svc.fit(X_train,y_train)

print(f'polynomial_svc학습 데이터 점수 : {polynomial_svc.score(X_train, y_train)}')
print(f'polynomial_svc평가 데이터 점수 : {polynomial_svc.score(X_test, y_test)}')

rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(X_train,y_train)

print(f'rbf_svc학습 데이터 점수 : {rbf_svc.score(X_train, y_train)}')
print(f'rbf_svc평가 데이터 점수 : {rbf_svc.score(X_test, y_test)}')


#########매개변수 튜닝
# SVM은 사용하는 커널에 따라 다양한 매개변수 설정 가능
# 매개변수(하이퍼 파라미터)를 변경하면서 성능변화를 관찰 가능

# degree가 무엇인가?? 차수
# C는 무엇인가?????
# gamma는 무엇인가?????

X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

polynomial_svc = SVC(kernel='poly',degree=2, C=0.1, gamma='auto')
polynomial_svc.fit(X_train,y_train)

print(f"polynomial_svc(dgree=2, C=0.1, gmma='auto')학습 데이터 점수 : {polynomial_svc.score(X_train, y_train)}")
print(f"polynomial_svc(dgree=2, C=0.1, gmma='auto')평가 데이터 점수 : {polynomial_svc.score(X_test, y_test)}")


rbf_svc = SVC(kernel='rbf', C=2.0, gamma='scale')
rbf_svc.fit(X_train,y_train)

print(f"rbf_svc(C=2.0, gamma='scale')학습 데이터 점수 : {rbf_svc.score(X_train, y_train)}")
print(f"rbf_svc(C=2.0, gamma='scale')평가 데이터 점수 : {rbf_svc.score(X_test, y_test)}")

#########데이터 전처리
# SVM은 입력데이터가 정규화 되어야 좋은 성능을 보임
# 주로 모든 특성값을 [0,1] 범위로 맞추는 방법을 사용
# scikit-learn의 StandardScaler 또는 MinMaxScaler을 사용하여 정규화


X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

model = SVC()
model.fit(X_train,y_train)

print(f'학습 데이터 점수 : {model.score(X_train, y_train)}')

print(f'평가 데이터 점수 : {model.score(X_test, y_test)}')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


model = SVC()
model.fit(X_train,y_train)

print(f'학습(StandardScaler) 데이터 점수 : {model.score(X_train, y_train)}')
print(f'평가(StandardScaler) 데이터 점수 : {model.score(X_test, y_test)}')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


model = SVC()
model.fit(X_train,y_train)

print(f'학습(MinMaxScaler) 데이터 점수 : {model.score(X_train, y_train)}')
print(f'평가(MinMaxScaler) 데이터 점수 : {model.score(X_test, y_test)}')

# StandardScaler는 SND (Standard Normal Distribution)를 따릅니다. 따라서 평균을 0으로 만들고 데이터를 단위 분산으로 조정합니다.
# MinMaxScaler는 [0, 1] 범위 또는 데이터 세트에 음수 값이있는 경우 [-1, 1] 범위의 모든 데이터 특성을 조정합니다. 
# 이 스케일링은 좁은 범위 [0, 0.005]의 모든 inliers를 압축합니다.
# 특이치가있는 경우 StandardScaler는 경험적 평균과 표준 편차를 계산하는 동안 특이치의 영향으로 균형 잡힌 특성 척도를 보장하지 않습니다. 이로 인해 기능값 범위가 축소됩니다.

# C는 데이터 샘플들이 다른 클래스에 놓이는 것을 허용하는 정도를 결정하고, 
# gamma는 결정 경계의 곡률을 결정한다. 





####Linear SVR(보스턴)
X,y = load_boston(return_X_y= True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = SVR(kernel='linear')
model.fit(X_train, y_train)
print(f'학습 데이터 점수 : {model.score(X_train, y_train)}')
print(f'평가 데이터 점수 : {model.score(X_test, y_test)}')
#저차원으로 변환
X_comp = TSNE(n_components=1).fit_transform(X)

plt.scatter(X_comp, y)
plt.show()

model.fit(X_comp,y)
predict = model.predict(X_comp)
plt.scatter(X_comp,y)
plt.scatter(X_comp, predict, color='r')
plt.show()

estimator = make_pipeline(StandardScaler(), SVR(kernel= 'linear'))

cross_validate(
    estimator =estimator,
    X=X, y=y,
    cv = 5,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)

pipe = Pipeline([('scaler', StandardScaler()),
                ('model', SVR(kernel='linear'))])
param_grid = [{'model__gamma' : ['scale','auto'],
                'model__C' : [1.0,0.1,0.001],
                'model__epsilon' : [1.0,0.1,0.001]}]

gs = GridSearchCV(
    estimator = pipe,
    param_grid=param_grid,
    n_jobs=multiprocessing.cpu_count(),
    cv=5, verbose = True
)
gs.fit(X,y)

print(gs.best_estimator_)


######Kernel SVR
#보스턴 주택 가격


X,y = load_boston(return_X_y= True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model = SVR(kernel='rbf')
model.fit(X_train, y_train)
print(f'학습 데이터 점수 : {model.score(X_train, y_train)}')
print(f'평가 데이터 점수 : {model.score(X_test, y_test)}')

X_comp = TSNE(n_components=1).fit_transform(X)
plt.scatter(X_comp, y)
plt.show()

model.fit(X_comp,y)
predict = model.predict(X_comp)
plt.scatter(X_comp,y)
plt.scatter(X_comp, predict, color='r')
plt.show()

estimator = make_pipeline(StandardScaler(), SVR(kernel= 'rbf'))
cross_validate(
    estimator =estimator,
    X=X, y=y,
    cv = 5,
    n_jobs=multiprocessing.cpu_count(),
    verbose=True
)
pipe = Pipeline([('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf'))])
param_grid = [{'model__gamma' : ['scale','auto'],
                'model__C' : [1.0,0.1,0.001],
                'model__epsilon' : [1.0,0.1,0.001]}]
gs = GridSearchCV(
    estimator = pipe,
    param_grid=param_grid,
    n_jobs=multiprocessing.cpu_count(),
    cv=5, verbose = True
)
gs.fit(X,y)
print(gs.best_estimator_)


