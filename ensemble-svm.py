# -*- coding: utf-8 -*-
 
import random
from sklearn.svm import SVC
import os 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics  
from pandas import *
import pandas as pd
import numpy as np 

#训练集和验证集
X_test = pd.read_table('D:\X_test.csv',encoding='gb2312',sep=",")          
X_train = pd.read_table('D:\X_train.csv',encoding='gb2312',sep=",")          
y_test = pd.read_table('D:\y_test.csv',encoding='gb2312',sep=",",header=None)          
y_train = pd.read_table('D:\y_train.csv',encoding='gb2312',sep=",",header=None)          

#训练集和测试集

def pipeline(iteration,C,gamma,random_seed):
    X_train_sub, _x , y_train_sub, _y = train_test_split(X_train,y_train,test_size=0.4,random_state=random_seed)
    print (X_train_sub.shape)
    clf = SVC(C=C,kernel='rbf',gamma=gamma,probability=True,cache_size=700,class_weight='balanced',verbose=True,random_state=random_seed)
    clf.fit(X_train_sub,y_train_sub)
    #输出概率
    pred = clf.predict_proba(X_test)
    test_result = pd.DataFrame(columns=["score"])
    test_result.score = pred[:,1]
    test_result.to_csv('D:/val2/svm_{0}.csv'.format(iteration),index=None)
    #输出二分类结果
    pred_C = clf.predict(X_test)
    test_result_C = pd.DataFrame(columns=["score"])
    test_result_C.score = pred_C
    test_result_C.to_csv('D:/val2/svm_C{0}.csv'.format(iteration),index=None)
    
random_seed = range(2016,2046)
C = [i/10.0 for i in range(10,40)]
gamma = [i/1000.0 for i in range(1,31)]
random.shuffle(list(random_seed))
print(random_seed)
random.shuffle(list(C))
print(C)
random.shuffle(list(gamma))
print(gamma)

#训练集和验证集
for i in range(30):
    pipeline(i,C[i],gamma[i],random_seed[i])
    #计算每个模型的auc
files = os.listdir('D:/val7')
for f in files[0:]:
    pred = pd.read_csv('D:/val7/'+f)
    score = pred.score
    auc = roc_auc_score(y_test,score)
    print(auc)

##计算概率均值的AUC
files = os.listdir('D:/val7')
pred = pd.read_csv('D:/val7/'+files[0])
score = pred.score
for f in files[1:]:
    pred = pd.read_csv('D:/val7/'+f)
    score += pred.score
score /= len(files)
auc = roc_auc_score(y_test,score)  #0.67643962452394457

###rank 算融合的probalitity

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
    
#生成报告
from sklearn.metrics import classification_report
print(classification_report(y_test, yu))
#混淆矩阵
from sklearn.metrics import confusion_matrix
m = confusion_matrix(y_test, yu) 
m 
plt.figure(figsize=(5,3))
sns.heatmap(m) 
plt.savefig("D:/confusion matrix under RF.jpg", dpi=300)
    