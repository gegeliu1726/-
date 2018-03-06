# -*- coding: utf-8 -*-

# xgboost选择特征
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import matplotlib.pyplot as plt  

model = XGBClassifier()
model.fit(X, y)
ax = plot_importance(model)
fig = ax.figure
fig.set_size_inches(15, 10)
plt.title('Feature under xgboostClassifier', y=1.05, size=15)
plt.savefig("Feature under XGB.jpg", dpi=300)
pyplot.show()

drop_col = ['has_fund_Unknown','marry_status_widowed','ln_sql_other',
            'ln_sql_later','ln_sql_replace','ovd_2_ln',
            'marry_status_other','NMG',
            'edu_level_master','edu_level_other'
            ]
col_new = X_train.columns.drop(drop_col) #剔除冗余特征
X_train_sel = X[col_new]
X_validation_sel = X_validation[col_new]

## xgboost
model = XGBClassifier()
eval_set = [(X_validation_sel, y_validation)]
model.fit(X_train_sel, y, early_stopping_rounds=50, 
          eval_metric="auc", eval_set=eval_set, verbose=True)

model = XGBClassifier(objective='binary:logistic',
                      colsample_bytree=0.9,
                      subsample=0.7,
                      max_depth=6,
                      reg_lambda=1,
                      min_child_weight=1,
                      n_estimators=300,
                      learning_rate=0.06,
                      seed=2018
                      )
eval_set = [(X_validation_sel, y_validation)]
model.fit(X_train_sel, y, early_stopping_rounds=50, 
          eval_metric="auc", eval_set=eval_set, verbose=True)
#0.75313
predicted = model.predict(X_validation_sel) # 通过分类器产生预测结果
probability = model.predict_proba(X_validation_sel)[:,1]
from sklearn import metrics
print("Validation set accuracy score: {:.5f}".format(metrics.accuracy_score(y_validation, predicted)))
print("Validation set AUC score: {:.5f}".format(metrics.roc_auc_score(y_validation, probability)))
#生成报告
from sklearn.metrics import classification_report
print(classification_report(y_validation, predicted))
#混淆矩阵
from sklearn.metrics import confusion_matrix
m = confusion_matrix(y_validation, predicted) 
m 
plt.figure(figsize=(5,3))
sns.heatmap(m) 
plt.savefig("confusion matrix under xgboost.jpg", dpi=300)

y_pred_df = pd.DataFrame(columns=['y_pred'])
y_pred_df.y_pred = predicted
y_pred_df.to_csv('y_pred_xgboost.csv',index = False)

y_proba_df = pd.DataFrame(columns=['y_proba'])
y_proba_df.y_proba = probability
y_proba_df.to_csv('y_proba_xgboost.csv',index = False)

#调参
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

model = XGBClassifier(objective='binary:logistic',  
                      learning_rate=0.06,
                      max_depth=6,
                      seed=2018,
                      colsample_bytree=0.9,
                      n_estimators=500)
param_grid1 = {'learning_rate':[0.25, 0.3, 0.35]}
param_grid2 = {'max_depth':[4, 6, 8],
              'min_child_weight':[1, 3, 5]
             }
param_grid3 = {'max_depth':[5, 6, 7],
              'subsample':[0.85, 0.9, 0.95]
             }
param_grid4 = {'colsample_bytree':[0.7, 0.8, 0.9],
               'subsample':[0.8, 0.85], 
               'max_depth':[8, 9]}
param_grid5 = {'gamma':[i/10.0 for i in range(0,5)]}
param_grid6 = {'subsample':[i/100.0 for i in range(70,95,5)],
               'colsample_bytree':[i/100.0 for i in range(70,95,5)]}
param_grid7 = {'subsample':[i/100.0 for i in range(65,80,2)]}
param_grid8 = {'learning_rate':[i/50.0 for i in range(1,20)]}
grid_search = GridSearchCV(model, param_grid7, scoring="roc_auc",
                           iid=False, cv=10)
grid_search.fit(X_train_sel, y)
print(grid_search.best_params_)
y_true, y_pred = y_validation, grid_search.predict(X_validation_sel)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred)) 
y_proba=grid_search.predict_proba(X_validation_sel)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_true, y_proba)) 