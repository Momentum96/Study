from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('feature들의 평균')
print(iris_df.mean())
print('\nfeature들의 분산')
print(iris_df.var())

from sklearn.preprocessing import StandardScaler # 평균 0, 분산 1에 가깝게

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature들의 평균')
print(iris_df_scaled.mean())
print('\nfeature들의 분산')
print(iris_df_scaled.var())