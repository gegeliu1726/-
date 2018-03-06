# -*- coding: utf-8 -*-

import sklearn
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np

data = pd.read_csv('train.csv', encoding='gb2312')
X_data, y_data = data.iloc[:,1:66], data.iloc[:,66]

# 数据标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_data_std = sc.fit_transform(X_data)
X_data_std = pd.DataFrame(X_data_std, columns=X_data.columns)

# 划分训练测试集
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_data_std, y_data, 
                                                    test_size=0.25, 
                                                    random_state=2018)

# rfe递归特征消除
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 1)
rfe = rfe.fit(X_train, y_train)
col = X_train.columns
rank = rfe.ranking_
print('Feature sorted by the rank:')
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), col)))

# pearson相关系数
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style('whitegrid')
colormap = plt.cm.viridis
plt.figure(figsize=(30,30))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X_train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.savefig("Feature correlation.jpg", dpi=300)

# 训练集smote
n_sample = y_train.shape[0]
n_pos_sample = y_train[y_train == 0].shape[0]
n_neg_sample = y_train[y_train == 1].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
print('特征维数：', X_train.shape[1])

from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
sm = SMOTE(random_state=2018)    # 处理过采样的方法
X, y = sm.fit_sample(X_train, y_train)
print('通过SMOTE方法平衡正负样本后')
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]
n_neg_sample = y[y == 1].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))





























