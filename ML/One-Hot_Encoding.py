from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

items=['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

labels = labels.reshape(-1, 1)
print(labels)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print("원-핫 인코더 데이터 차원")
print(oh_labels.shape)

# Pandas OneHotEncoding API

import pandas as pd

df = pd.DataFrame({'item':items})
print(pd.get_dummies(df))