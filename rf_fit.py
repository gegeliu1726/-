# -*- coding: utf-8 -*-
import sklearn
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np


# 随机森林选择特征
X = pd.DataFrame(X, columns=X_data.columns)
features1 = X.columns
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=500,random_state=2018)#构建分类随机森林分类器
clf.fit(X, y) #对自变量和因变量进行拟合
features1, clf.feature_importances_
importance = np.argsort(clf.feature_importances_)[::-1]
features_sorted, importances_sorted = features1[importance], clf.feature_importances_[importance]
for feature in zip(features_sorted, importances_sorted):
    print(feature)

plt.bar(range(len(features_sorted)), importances_sorted)
plt.xticks(range(len(features_sorted)), features_sorted, rotation = 90)
plt.title('Feature under RandomForestClassifier', y=1.05, size=15)
plt.tight_layout()
plt.savefig("Feature under RFC.jpg", dpi=300)
plt.show


## 随机森林
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=500, 
                                random_state=2018, n_jobs=2)
forest.fit(X_train_sel, y)
predicted = forest.predict(X_validation_sel) # 通过分类器产生预测结果
probability = forest.predict_proba(X_validation_sel)[:,1]
from sklearn import metrics
print("Validation set accuracy score: {:.5f}".format(metrics.accuracy_score(y_validation, predicted)))
print("Validation set AUC score: {:.5f}".format(metrics.roc_auc_score(y_validation, probability)))
#0.93947 0.73593
#生成报告
from sklearn.metrics import classification_report
print(classification_report(y_validation, predicted))
#混淆矩阵
from sklearn.metrics import confusion_matrix
m = confusion_matrix(y_validation, predicted) 
m 
plt.figure(figsize=(5,3))
sns.heatmap(m) 
plt.savefig("confusion matrix under RF.jpg", dpi=300)

y_pred_df = pd.DataFrame(columns=['y_pred'])
y_pred_df.y_pred = predicted
y_pred_df.to_csv('y_pred_RF.csv',index = False)

y_proba_df = pd.DataFrame(columns=['y_proba'])
y_proba_df.y_proba = probability
y_proba_df.to_csv('y_proba_RF.csv',index = False)



