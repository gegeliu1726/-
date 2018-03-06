# -*- coding: utf-8 -*-


#导入、处理测试集
import sklearn
import pydotplus
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
new_data = pd.read_csv('new_test.csv', encoding='gb2312')
new_test = new_data.iloc[:,1:66]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
new_test_std = sc.transform(new_test)

drop_col = ['has_fund_Unknown','marry_status_widowed','ln_sql_other',
            'ln_sql_later','ln_sql_replace','ovd_2_ln',
            'marry_status_other','NMG',
            'edu_level_master','edu_level_other'
            ]
col_new = new_test_std.columns.drop(drop_col) #剔除冗余特征
new_test_sel = new_test_std[col_new]
new_test_sel.to_csv('D://new_test_sel.csv',index = False)

#随机森林预测
predicted = forest.predict(new_test_sel) # 通过分类器产生预测结果
probability = forest.predict_proba(new_test_sel)[:,1]

y_pred_df = pd.DataFrame(columns=['y_pred'])
y_pred_df.y_pred = predicted
y_pred_df.to_csv('new_pred_RF.csv',index = False)

y_proba_df = pd.DataFrame(columns=['y_proba'])
y_proba_df.y_proba = probability
y_proba_df.to_csv('new_proba_RF.csv',index = False)

#xgboost预测
predicted = model.predict(new_test_sel) # 通过分类器产生预测结果
probability = model.predict_proba(new_test_sel)[:,1]

y_pred_df = pd.DataFrame(columns=['y_pred'])
y_pred_df.y_pred = predicted
y_pred_df.to_csv('new_pred_xgboost.csv',index = False)

y_proba_df = pd.DataFrame(columns=['y_proba'])
y_proba_df.y_proba = probability
y_proba_df.to_csv('new_proba_xgboost.csv',index = False)

