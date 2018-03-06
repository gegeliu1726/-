###"东证期货杯"全国统计建模大赛的提交文件说明

- 2+976+15827592612.pdf：PDF格式的初赛正文论文

- test_answer.xlsx：测试集预测出的结果

- code文件夹：
  - feature_create.py 根据原始特征衍生新特征

  - smote.py  对原始数据进行标准化，进行 SMOTE采样

  - visualize_dataset.py 缺失值的可视化分析以及一些探索性分析的作图

  - rf_fit.py    训练随机森林模型

  - xgb_fit.py 训练模型xgboost模型，并利用网格搜索进行调参

  - rf_xgb_predict.py  利用训练出的模型对从训练集划分出的验证集进行预测

    利用rf模型得到的auc值是0.73593左右

    利用xgboost模型得到的auc值是0.75313左右

  - ensemble-svm.py  运用数据集分解的方法训练30个svm进行averaging，得到的auc值是0.723左右

  - rank_weight.py 将训练出来的xgboost、随机森林以及向量机模型的概率值进行加权 ，权重分别为0.6,0.2,0.2，得到的auc值是0.758左右
