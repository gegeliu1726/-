# -*- coding: utf-8 -*-


###rank 算融合的probalitity
#训练集和验证集
files = os.listdir('D:/all')
rf = pd.read_csv('D:/all/y_proba_RF.csv')
svm = pd.read_csv('D:/all/y_proba_svm.csv')
xgboost = pd.read_csv('D:/all/y_proba_xgboost.csv')
rf.score = rf.score.rank()
svm.score = svm.score.rank()
xgboost.score = xgboost.score.rank()
pred = 0.6*xgboost.score + 0.2*rf.score + 0.2*svm.score
roc_auc_score(y_test,pred)
#0.757966490002836

#训练集和测试集
files = os.listdir('D:/all')
rf = pd.read_csv('D:/all/y_proba_RF.csv')
svm = pd.read_csv('D:/all/y_proba_svm.csv')
xgboost = pd.read_csv('D:/all/y_proba_xgboost.csv')
rf.score = rf.score.rank()
svm.score = svm.score.rank()
xgboost.score = xgboost.score.rank()
pred = 0.6*xgboost.score + 0.2*rf.score + 0.2*svm.score
roc_auc_score(y_test,pred)
finall=pd.DataFrame(pred)
pred=pred/10000
pred_1=np.array(pred)
yuzhi=np.int64(pred_1 > 0.65) #大于阈值的输出为1
yu = pd.Series(yuzhi)
accuracy = metrics.accuracy_score(y_test,yu)
print(accuracy)    