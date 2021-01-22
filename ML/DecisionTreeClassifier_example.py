from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

iris = load_iris() # 붓꽃 데이터 세트 로드

iris_data = iris.data # 입력 피처만으로 된 numpy 데이터

iris_label = iris.target # 예측 label
print('iris target값:', iris_label)
print('iris target명:', iris.target_names)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names) # dataframe 생성
iris_df['label'] = iris.target
print(iris_df.head(3))

print(iris_df.corr())

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=1) # 학습 데이터, 테스트 데이터 분리, feature들을 x, label을 y로, 80% 학습 데이터, 20% 테스트 데이터

dt_clf = DecisionTreeClassifier(random_state=1) # DecisionTreeClassifier 객체 생성
dt_clf.fit(x_train, y_train) # 모델 학습

pred = dt_clf.predict(x_test) # 모델 예측

from sklearn.metrics import accuracy_score # 정확도 파악
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

"""
1. 데이터 세트 분리 (학습 데이터, 테스트 데이터 분리)
2. 모델 학습
3. 예측 수행
4. 평가
"""