# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from pandas import *
import pandas as pd
import numpy as np 
import re

####导入各表格
df= pd.read_table('D:\contest_basic_test.tsv',encoding='gb2312',sep="\t")             #测试集
df1 = pd.read_table('D:\contest_basic_train.tsv',encoding='gb2312',sep="\t")          #训练集
df2 = pd.read_table('D:\contest_ext_crd_cd_ln.tsv',encoding='gb2312',sep="\t")        #贷款
df3 = pd.read_table('D:\contest_ext_crd_cd_lnd.tsv',encoding='gb2312',sep="\t")        #贷记卡
#df4 = pd.read_table('D:\contest_ext_crd_hd_report.csv',encoding='gb2312',sep=",")     #主表
#df5 = pd.read_table('D:\contest_ext_crd_cd_lnd_ovd.csv',encoding='gb2312',sep=",")    #贷记卡逾期记录
df6 = pd.read_table('D:\contest_ext_crd_is_ovdsummary.csv',encoding='gb2312',sep=",") #贷记卡和贷款逾期记录汇总
df7 = pd.read_table('D:\contest_ext_crd_cd_ln_spl.tsv',encoding='gb2312',sep="\t")    #特殊交易记录
df8 = pd.read_table('D:\contest_ext_crd_is_sharedebt.csv',encoding='utf-8',sep=",")  #未销户记录
df9 = pd.read_table('D:\contest_ext_crd_is_creditcue.csv',encoding='gb2312',sep=",")  #信用提示
df10 = pd.read_table('D:\contest_ext_crd_qr_recorddtlinfo.tsv',encoding='gb2312',sep="\t")#信贷审批查询记录

####处理特殊交易信息
df7 = pd.read_table('D:\contest_ext_crd_cd_ln_spl.tsv',encoding='gb2312',sep="\t")    #特殊交易记录
len(df7.REPORT_ID.unique()) ###这个表有14190人
df7.index=df7.REPORT_ID
df7.info()
df7.head(20)
df7_early=df7[(df7['type_dw']=='提前还款')]  
df7_early=df7_early.drop('content',axis=1)
early=df7_early.groupby('REPORT_ID').size()
df7_later=df7[(df7['type_dw']=='展期（延期）')]  
later=df7_later.groupby('REPORT_ID').size()
df7_daiti=df7[(df7['type_dw']=='担保人代还')]  
daiti=df7_later.groupby('REPORT_ID').size()
df7_other=df7[(df7['type_dw']=='其他')]  
other=df7_later.groupby('REPORT_ID').size()
#训练集
df1 = pd.read_table('D:\contest_basic_train.tsv',encoding='gb2312',sep="\t")          #训练集
df1.index=df1.REPORT_ID
df1['early']=0
df1['early']=early
df1['early']=df1['early'].fillna(0)
df1['later']=0
df1['later']=later
df1['later']=df1['later'].fillna(0)
df1['daiti']=0
df1['daiti']=daiti
df1['daiti']=df1['daiti'].fillna(0)
df1['other']=0
df1['other']=other
df1['other']=df1['other'].fillna(0)
df1.to_csv('D://train16.csv',index = False)

#测试集
df= pd.read_table('D:\contest_basic_test.tsv',encoding='gb2312',sep="\t")             #测试集
df.index=df.REPORT_ID
df['early']=0
df['early']=early
df['early']=df['early'].fillna(0)
df['later']=0
df['later']=later
df['later']=df['later'].fillna(0)
df['daiti']=0
df['daiti']=daiti
df['daiti']=df['daiti'].fillna(0)
df['other']=0
df['other']=other
df['other']=df['other'].fillna(0)
df.to_csv('D://test1.csv',index = False)


####信用历史（按月份计算）+房贷+商业贷款+其他贷款
df9 = pd.read_table('D:\contest_ext_crd_is_creditcue.csv',encoding='gb2312',sep=",")  #信用提示
df9.FIRST_LOAN_OPEN_MONTH=df9.FIRST_LOAN_OPEN_MONTH.replace('--','2018.01').astype('float') #无信用历史
df9.FIRST_LOANCARD_OPEN_MONTH=df9.FIRST_LOANCARD_OPEN_MONTH.replace('--','2018.01').astype('float')
df9.FIRST_SL_OPEN_MONTH=df9.FIRST_SL_OPEN_MONTH.replace('--','2018.01').astype('float')
df9['history']=0
a=df9.FIRST_LOAN_OPEN_MONTH
b=df9.FIRST_LOANCARD_OPEN_MONTH
c=df9.FIRST_SL_OPEN_MONTH
newdf9=pd.DataFrame([a,b,c]).T
newdf9.head()
newdf9['history']=np.min(newdf9,axis=1)
newdf9['REPORT_ID']=df9['REPORT_ID']
newdf9.shape
newdf9.index=newdf9.REPORT_ID
history_length=newdf9.history

#训练集
df1['history']=0
df1['history']=history_length
df1.history=df1.history.fillna(0)
len(df1.history[df1.history.isnull().values==True]) #找到缺失值的位置
df1_history_year=df1.history.astype('int') #年份
df1_load_year=df1.LOAN_DATE.str[:4] #年份
df1['length']=0
df1['length']=(df1_load_year.astype('int')-df1_history_year)*12-(df1.history-df1_history_year)*100+df1.LOAN_DATE.str[5:6].astype('int')
df1['length'].max()
df1['length'].min()
df1.length.describe()
df1['length'].sort_values() 
df1.length[df1['length']<0]=0 #无信用历史替换值
df1.length[df1['length']>20000]=np.NaN #信用历史缺失
len(df1.length[df1.length.isnull().values==True]) #找到缺失值的位置 最后有29个人信用历史缺失
df1.head()
cre_save=['REPORT_ID','HOUSE_LOAN_COUNT','COMMERCIAL_LOAN_COUNT','OTHER_LOAN_COUNT']
df9=df9[cre_save]
df.head()
df9.index=df9.REPORT_ID
df1[['HOUSE_LOAN_COUNT','COMMERCIAL_LOAN_COUNT','OTHER_LOAN_COUNT']]=df9[['HOUSE_LOAN_COUNT','COMMERCIAL_LOAN_COUNT','OTHER_LOAN_COUNT']]
df1.to_csv('D://train2.csv',index = False)

#测试集
df['history']=0
df['history']=history_length
df.history=df.history.fillna(0)
len(df.history[df.history.isnull().values==True]) #找到一个缺失值 8540113
df_history_year=df.history.astype('int') #年份
df_load_year=df.LOAN_DATE.str[:4] #年份
df['length']=0
df['length']=(df_load_year.astype('int')-df_history_year)*12-(df.history-df_history_year)*100+df.LOAN_DATE.str[5:6].astype('int')
df['length'].max()
df['length'].min()
df.length.describe()
df.length[df['length']<0]=0 #无信用历史
df.length[df['length']>20000]=np.NaN #信用历史缺失
len(df.length[df.length.isnull().values==True]) #找到缺失值的位置 最后有29个人信用历史缺失
df.length=df.length.fillna(0)
df[['HOUSE_LOAN_COUNT','COMMERCIAL_LOAN_COUNT','OTHER_LOAN_COUNT']]=df9[['HOUSE_LOAN_COUNT','COMMERCIAL_LOAN_COUNT','OTHER_LOAN_COUNT']]
df.to_csv('D://test3.csv',index = False)


####未销户信息
#未结清贷款信息
df8 = pd.read_table('D:\contest_ext_crd_is_sharedebt.csv',encoding='utf-8',sep=",")  #未销户记录
df8.columns
df8.shape
df8.head()
df8.index=df8.REPORT_ID
newdf8=df8[df8['TYPE_DW']=='未结清贷款信息汇总']  ##贷款笔数
sum_org_ln=newdf8.groupby('REPORT_ID').FINANCE_ORG_COUNT.sum()  #贷款机构数

#训练集
df1['sum_org_ln']=0
df1['sum_org_ln']=sum_org_ln
df1.to_csv('D://train13.csv',index = False)

#测试集
df['sum_org_ln']=0
df['sum_org_ln']=sum_org_ln
df.sum_org_ln=df.sum_org_ln.fillna(0)
df.to_csv('D://test4.csv',index = False)

newdf8.columns
save_column=['FINANCE_ORG_COUNT','CREDIT_LIMIT','BALANCE','LATEST_6M_USED_AVG_AMOUNT']
newdf8_save=newdf8[save_column]
newdf8_save.head()

#训练集
df1.index=df1.REPORT_ID
df1[['FINANCE_ORG_COUNT','CREDIT_LIMIT','BALANCE','LATEST_6M_USED_AVG_AMOUNT']]=newdf8_save
df1.head()
df1.columns
df1=df1.fillna(0)
df1.to_csv('D://train4_2.csv',index = False)

#测试集
df.index=df.REPORT_ID
df[['FINANCE_ORG_COUNT','CREDIT_LIMIT','BALANCE','LATEST_6M_USED_AVG_AMOUNT']]=newdf8_save
df.head()
df.columns
df.FINANCE_ORG_COUNT=df.FINANCE_ORG_COUNT.fillna(0)
df.CREDIT_LIMIT=df.CREDIT_LIMIT.fillna(0)
df.BALANCE=df.BALANCE.fillna(0)
df.LATEST_6M_USED_AVG_AMOUNT=df.LATEST_6M_USED_AVG_AMOUNT.fillna(0)

###为结清贷记卡和准贷记卡信息
df8 = pd.read_table('D:\contest_ext_crd_is_sharedebt.csv',encoding='utf-8',sep=",")  #未销户记录
newdf8_2=df8[(df8['TYPE_DW']=='未销户贷记卡信息汇总') | (df8['TYPE_DW']=='未销户准贷记卡信息汇总')]  ##贷款笔数
newdf8_2.shape
newdf8_2.head(60)
save_column=['REPORT_ID','TYPE_DW','FINANCE_ORG_COUNT','CREDIT_LIMIT','MAX_CREDIT_LIMIT_PER_ORG','MIN_CREDIT_LIMIT_PER_ORG','USED_CREDIT_LIMIT','LATEST_6M_USED_AVG_AMOUNT']
newdf8_save2=newdf8_2[save_column]
newdf8_save2.head()
newdf8_save2.index=newdf8_save2.REPORT_ID
#贷、准记卡机构数之和
sum_org=newdf8_save2.groupby('REPORT_ID').FINANCE_ORG_COUNT.sum() 
#贷、准记卡合同金额之和
sum_credit=newdf8_save2.groupby('REPORT_ID').CREDIT_LIMIT.sum()   
#贷、准记卡单笔机构合同金额最大值
max_credit=newdf8_save2.groupby('REPORT_ID').MAX_CREDIT_LIMIT_PER_ORG.max() 
#贷、准记卡单笔机构合同金额最小值
min_credit=newdf8_save2.groupby('REPORT_ID').MIN_CREDIT_LIMIT_PER_ORG.min() 
#贷、准记卡最近6个月平均消费金额
sum_6m_uesd_avg=newdf8_save2.groupby('REPORT_ID').LATEST_6M_USED_AVG_AMOUNT.sum()
#贷、准记卡已用额度之和
sum_used_credit=newdf8_save2.groupby('REPORT_ID').USED_CREDIT_LIMIT.sum()
newdf8_3=pd.DataFrame([sum_org,sum_credit,max_credit,min_credit,sum_used_credit,sum_6m_uesd_avg]).T
newdf8_3.head()

#训练集
df1 = pd.read_table('D:\contest_basic_train.tsv',encoding='gb2312',sep="\t")          #训练集
df1.index=df1.REPORT_ID
df1[['FINANCE_ORG_COUNT','CREDIT_LIMIT','MAX_CREDIT_LIMIT_PER_ORG','MIN_CREDIT_LIMIT_PER_ORG','USED_CREDIT_LIMIT','LATEST_6M_USED_AVG_AMOUNT']]=newdf8_3
df1.head()
df1.columns
df1.to_csv('D://train5.csv',index = False)

#测试集
df= pd.read_table('D:\contest_basic_test.tsv',encoding='gb2312',sep="\t")             #测试集
df.index=df.REPORT_ID
df[['FINANCE_ORG_COUNT','CREDIT_LIMIT','MAX_CREDIT_LIMIT_PER_ORG','MIN_CREDIT_LIMIT_PER_ORG','USED_CREDIT_LIMIT','LATEST_6M_USED_AVG_AMOUNT']]=newdf8_3
df=df.fillna(0)
df.head(20)
df.columns
df.to_csv('D://test6.csv',index = False)


####被查询次数
#信用卡和贷款的查询次数
df10 = pd.read_table('D:\contest_ext_crd_qr_recorddtlinfo.tsv',encoding='gb2312',sep="\t")#信贷审批查询记录
df10_lnd=df10[(df10['query_reason']=='信用卡审批')|(df10['query_reason']=='贷款审批')]  
df10.info()
df10.head()
df10_lnd.head()
query_times=df10_lnd.groupby('REPORT_ID').size()

#训练集
df1.index=df1.REPORT_ID
df1['query_times']=0
df1['query_times']=query_times 
df1.to_csv('D://train13.csv',index = False)

#测试集
df.head()
df['query_times']=0
df['query_times']=query_times 
df['query_times']=df['query_times'].fillna(0)
df.to_csv('D://test7.csv',index = False)


####担保资格审查查询次数
df10_danbao=df10[(df10['query_reason']=='担保资格审查')]  
danao_record=df10_danbao.groupby('REPORT_ID').size() 
#训练集
df1['danao_record']=0
df1['danao_record']=danao_record 
len(df1.danao_record[df1.danao_record.isnull().values==True]) #在2014年之后有24633人没有贷款查询记录
df1.head()
df1.to_csv('D://train8.csv',index = False)
#测试集
df['danao_record']=0
df['danao_record']=danao_record 
len(df.danao_record[df.danao_record.isnull().values==True]) #在2014年之后有24633人没有贷款查询记录
df['danao_record']=df['danao_record'].fillna(0)
df.head()
df.to_csv('D://test8.csv',index = False)


####逾期信息汇总
df6 = pd.read_table('D:\contest_ext_crd_is_ovdsummary.csv',encoding='gb2312',sep=",") #贷记卡和贷款逾期记录汇总
df6.info()
df6.head()
df6.index=df6.REPORT_ID
#贷款、准、贷记卡逾期笔数或者账户数总和
sum_count=df6.groupby('REPORT_ID').COUNT_DW.sum()
#贷款、准、贷记卡逾期月份数总和
sum_mon=df6.groupby('REPORT_ID').MONTHS.sum() 
#贷款、准、贷记卡单月最高逾期金额
max_mon_avg=df6.groupby('REPORT_ID').HIGHEST_OA_PER_MON.max()
#贷款、准、贷记卡最长逾期时间
max_duration=df6.groupby('REPORT_ID').MAX_DURATION.max()
newdf6=pd.DataFrame([sum_count,sum_mon,max_mon_avg,max_duration]).T
newdf6.head()

#训练集
df1 = pd.read_table('D:\contest_basic_train.tsv',encoding='gb2312',sep="\t")          #训练集
df1.index=df1.REPORT_ID
df1[['COUNT_DW','MONTHS','HIGHEST_OA_PER_MON','MAX_DURATION']]=newdf6
df1.head()
df1=df1.fillna(0)
df1.to_csv('D://train6.csv',index = False)

#测试集
df[['COUNT_DW','MONTHS','HIGHEST_OA_PER_MON','MAX_DURATION']]=newdf6
df.head()
df=df.fillna(0)
df.to_csv('D://test9.csv',index = False)


####贷款信息
df2 = pd.read_table('D:\contest_ext_crd_cd_ln.tsv',encoding='gb2312',sep="\t")        #贷款
len(df2.REPORT_ID.unique()) ###这个表有35594人
df2.info()
df2.shape
df2.head()
df2.index=df2.REPORT_ID
over_max=df2.groupby('REPORT_ID').curr_overdue_amount.max() #当前逾期金额取最大值
num=df2.groupby('REPORT_ID').size()  #贷款笔数
#对账户状态和五级分类进行编码
def coding(col,codeDict):
    colCoded = pd.Series(col,copy=True)
    for key,value in codeDict.items():
        colCoded.replace(key,value,inplace = True)
    return colCoded

df2['state']=coding(df2['state'], {'结清':0,'转出':0,'正常':1,'逾期':2,'呆账':3}) #worst_state
df2['class5_state']=coding(df2['class5_state'], {'未知':1,'正常':1,'关注':2,'次级':3,'可疑':4}) #
bad_state=df2.groupby('REPORT_ID').state.max()
bad_class5_state=df2.groupby('REPORT_ID').class5_state.max()

####处理24个月还款记录
#近一年内逾期1次的总数
df2_N=df2[(df2['state']=='正常')|(df2['state']=='逾期')] 
df2_N.shape #有167694条记录
report_id=df2_N.REPORT_ID
len(df2_N.REPORT_ID.unique()) ###这个表有33707人
col_1=['REPORT_ID','guarantee_type','payment_state']
df2_N_24=df2_N[col_1]
df2_N_24.index=df2_N_24.REPORT_ID
df2_N_24=df2_N_24.drop('REPORT_ID',axis=1)
df2_N_24.head()
payment_state=df2_N_24.payment_state.str.split()
payment_state.shape
new_pay=np.zeros(167694)
regex = re.compile('1')
for i in range(167694):
    res=regex.findall(payment_state.values[i][-10:]) #利用切片选出近12个月
    new_pay[i]=len(res)
    #print(new_pay[i])
pay=pd.Series(new_pay,index=report_id)
ovd_1_ln=pay.groupby('REPORT_ID').sum() #近一年逾期次数为1的总和
#ovd_1_ln.loc[2584739]
#ovd_1_ln[:50]

#两年内的超过一次逾期次数之和
new_pay_24=np.zeros(167694)
pattern=r'[234567]'
regex_24 = re.compile(pattern)
#test_1='##234##67##'
#res=regex_24.findall(test_1)
for i in range(167694):
    res=regex_24.findall(payment_state.values[i])
    new_pay_24[i]=len(res)
    #print(new_pay_24[i])
pay_24=pd.Series(new_pay_24,index=report_id)
ovd_2_ln=pay_24.groupby('REPORT_ID').sum() #两年内逾期次数超过一次的次数和
ovd_2_ln[:50]

#两年内正常还款的比例
new_pay_24_2=np.zeros(167694)
pattern_1=r'[NC]'
regex_24_2= re.compile(pattern_1)
for i in range(167694):
    res=regex_24_2.findall(payment_state.values[i])
    new_pay_24_2[i]=len(res)
    #print(new_pay_24[i])
pay_24_2=pd.Series(new_pay_24_2,index=report_id)

new_pay_24_3=np.zeros(167694)
pattern_2=r'[NC1234567]'
regex_24_3 = re.compile(pattern_2)
#test_2='NN##234##67##'
#res=regex_24_3.findall(test_2)
for i in range(167694):
    res=regex_24_3.findall(payment_state.values[i])
    new_pay_24_3[i]=len(res)
    #print(new_pay_24[i])
pay_24_3=pd.Series(new_pay_24_3,index=report_id)
pay_N_percent=pay_24_2/pay_24_3
pay_N=pay_N_percent.groupby('REPORT_ID').min() #取最小值

#贷款余额与贷款合同金额的平均比例
df2_N=df2[(df2['state']=='正常')|(df2['state']=='逾期')] 
df2_N.info()
df2_N.head()
df2_N.index=df2_N.REPORT_ID
balance = df2_N['balance']
credit= df2_N['credit_limit_amount']
balance_per=balance/credit
bal_per = balance_per.groupby('REPORT_ID').mean()
#针对担保方式设置一个指标
#信用免担保的贷款总笔数
df2_N_free=df2_N[df2_N['guarantee_type']=='信用/免担保']
df2_N_free.head(10)
num_guarantee_free_ln=df2_N_free.groupby('REPORT_ID').size()
#信用免担保贷款的平均合同额度
mean_guarantee_free_ln=df2_N_free.groupby('REPORT_ID').credit_limit_amount.mean()

#贷款的最长信用年限 天数为单位
df2_N['open_date'] = pd.to_datetime(df2_N['open_date'])
df2_N['end_date'] = pd.to_datetime(df2_N['end_date'])
credit_month = df2_N['end_date']-df2_N['open_date']
credit_month=credit_month.groupby('REPORT_ID').max()

#训练集
df1.index=df1.REPORT_ID
df1['over_max']=0
df1['over_max']=over_max
df1['num']=0
df1['num']=num
df1['bad_state']=0
df1['bad_state']=bad_state
df1['bad_class5_state']=0
df1['bad_class5_state']=bad_class5_state
df1['bad_class5_state'].fillna(1,inplace=True)
df1['ovd_1_ln']=0
df1['ovd_1_ln']=ovd_1_ln
df1['ovd_1_ln']=df1['ovd_1_ln'].fillna(0)
df1['ovd_1_ln'].head(50)
df1['ovd_2_ln']=0
df1['ovd_2_ln']=ovd_2_ln
df1['ovd_2_ln']=df1['ovd_2_ln'].fillna(0)
df1['ovd_2_ln'].head(50)
df1['pay_N_percent']=0
df1['pay_N_percent']=pay_N
df1['pay_N_percent']=df1['pay_N_percent'].fillna(0)
df1['bal_per']=0
df1['bal_per']=bal_per
df1['bal_per']=df1['bal_per'].fillna(0)
df1['num_guarantee_free_ln']=0
df1['num_guarantee_free_ln']=num_guarantee_free_ln
df1['num_guarantee_free_ln']=df1['num_guarantee_free_ln'].fillna(0)
df1['mean_guarantee_free_ln']=0
df1['mean_guarantee_free_ln']=mean_guarantee_free_ln
df1['mean_guarantee_free_ln']=df1['mean_guarantee_free_ln'].fillna(0)
df1['credit_month']=Timedelta('0 days 00:00:00')
df1['credit_month']=credit_month
df1['credit_month']=df1['credit_month'].fillna(Timedelta('0 days 00:00:00'))
df1.to_csv('D://train22.csv',index = False)

#测试集
df.index=df.REPORT_ID
df.head()
df['over_max']=0
df['over_max']=over_max
df['over_max']=df['over_max'].fillna(0)
df['num']=0
df['num']=num
df['num']=df['num'].fillna(0)
df['bad_state']=0
df['bad_state']=bad_state
df['bad_state'].fillna(0,inplace=True)
df['bad_class5_state']=0
df['bad_class5_state']=bad_class5_state
df['bad_class5_state'].fillna(1,inplace=True)
df['ovd_1_ln']=0
df['ovd_1_ln']=ovd_1_ln
df['ovd_1_ln']=df['ovd_1_ln'].fillna(0)
df['ovd_2_ln']=0
df['ovd_2_ln']=ovd_2_ln
df['ovd_2_ln']=df['ovd_2_ln'].fillna(0)
df['pay_N_per_ln']=0
df['pay_N_per_ln']=pay_N
df['pay_N_per_ln']=df['pay_N_per_ln'].fillna(0)
df['bal_per']=0
df['bal_per']=bal_per
df['bal_per']=df['bal_per'].fillna(0)
df['num_guarantee_free_ln']=0
df['num_guarantee_free_ln']=num_guarantee_free_ln
df['num_guarantee_free_ln']=df['num_guarantee_free_ln'].fillna(0)
df['mean_guarantee_free_ln']=0
df['mean_guarantee_free_ln']=mean_guarantee_free_ln
df['mean_guarantee_free_ln']=df['mean_guarantee_free_ln'].fillna(0)
df['credit_month']=Timedelta('0 days 00:00:00')
df['credit_month']=credit_month
df['credit_month']=df['credit_month'].fillna(Timedelta('0 days 00:00:00'))
#pay_N.loc[2298999]
#df['pay_N_per_ln'].loc[6468720]
df.to_csv('D:///test16.csv',index = False)



####贷记卡信息
df3 = pd.read_table('D:\contest_ext_crd_cd_lnd.tsv',encoding='gb2312',sep="\t") 
len(df3.REPORT_ID.unique()) ###这个表有39850人      
df3_lnd=df3[(df3['currency']=='人民币')]  
len(df3.REPORT_ID.unique()) ###这个表有39850人
lnd_num=df3.groupby('REPORT_ID').size()  ##贷记卡账户数
df3['state']=coding(df3['state'], {'正常':1,'销户':0,'未激活':0,'冻结':2,'止付':3,'呆账':4}) #worst_state
bad_state_lnd=df3.groupby('REPORT_ID').state.max()
over_max_lnd=df3.groupby('REPORT_ID').curr_overdue_amount.max()

#训练集
df1['bad_state_lnd']=0
df1['bad_state_lnd']=bad_state_lnd
df1['bad_state_lnd'].fillna(0,inplace=True)
df1['lnd_num']=0
df1['lnd_num']=lnd_num
df1['lnd_num'].fillna(0,inplace=True)
df1['over_max_lnd']=0
df1['over_max_lnd']=over_max_lnd
df1['over_max_lnd'].fillna(0,inplace=True)

#测试集
df['bad_state_lnd']=0
df['bad_state_lnd']=bad_state_lnd
df['bad_state_lnd'].fillna(0,inplace=True)
df['lnd_num']=0
df['lnd_num']=lnd_num
df['lnd_num'].fillna(0,inplace=True)
df['over_max_lnd']=0
df['over_max_lnd']=over_max_lnd
df['over_max_lnd'].fillna(0,inplace=True)
df.to_csv('D:///test11.csv',index = False)

####处理24个月还款记录
#近一年内贷记卡逾期1次的总数
df3.info()
df3_N=df3[(df3['state']=='正常')|(df3['state']=='冻结')|(df3['state']=='止付')] 
df3_N=df3_N[(df3_N['currency']=='人民币')] 
df3_N.shape #有175001条记录
report_id=df3_N.REPORT_ID
len(df3_N.REPORT_ID.unique()) ###这个表有39764人
col_2=['REPORT_ID','guarantee_type','payment_state']
df3_N_12=df3_N[col_2]
df3_N_12.index=df3_N_12.REPORT_ID
df3_N_12=df3_N_12.drop('REPORT_ID',axis=1)
df3_N_12.head()
payment_state=df3_N_12.payment_state
payment_state.shape
new_pay=np.zeros(175001)
regex = re.compile('1')
for i in range(175001):
    res=regex.findall(payment_state.values[i][-10:]) #利用切片选出近12个月
    new_pay[i]=len(res)
    #print(new_pay[i])
pay=pd.Series(new_pay,index=report_id)
ovd_1_lnd=pay.groupby('REPORT_ID').sum() #近一年逾期次数为1的总和
#ovd_1_ln.loc[2584739]
#ovd_1_ln[:50]

#两年内的超过一次逾期次数之和
new_pay_24=np.zeros(175001)
pattern=r'[234567]'
regex_24 = re.compile(pattern)
#test_1='##234##67##'
#res=regex_24.findall(test_1)
for i in range(175001):
    res=regex_24.findall(payment_state.values[i])
    new_pay_24[i]=len(res)
    #print(new_pay_24[i])
pay_24=pd.Series(new_pay_24,index=report_id)
ovd_2_lnd=pay_24.groupby('REPORT_ID').sum() #两年内逾期次数超过一次的次数和
ovd_2_lnd[:50]

#两年内正常还款的比例
new_pay_24_2=np.zeros(175001)
pattern_1=r'[NC]'
regex_24_2= re.compile(pattern_1)
for i in range(175001):
    res=regex_24_2.findall(payment_state.values[i])
    new_pay_24_2[i]=len(res)
    #print(new_pay_24[i])
pay_24_2=pd.Series(new_pay_24_2,index=report_id)

new_pay_24_3=np.zeros(175001)
pattern_2=r'[NC1234567]'
regex_24_3 = re.compile(pattern_2)
#test_2='NN##234##67##'
#res=regex_24_3.findall(test_2)
for i in range(175001):
    res=regex_24_3.findall(payment_state.values[i])
    new_pay_24_3[i]=len(res)
    #print(new_pay_24[i])
pay_24_3=pd.Series(new_pay_24_3,index=report_id)
pay_N_percent=pay_24_2/pay_24_3
pay_N=pay_N_percent.groupby('REPORT_ID').min() #取最小值

#贷记卡透支金额与贷记卡授信金额的平均比例
df8 = pd.read_table('D:\contest_ext_crd_is_sharedebt.csv',encoding='gb2312',sep=",")  #未销户记录
df8_N=df8[(df8['TYPE_DW']=='未销户贷记卡信息汇总')] 
df8.head()
df8.info()
df8_N.shape #有39831条记录
df8_N.index=df8_N.REPORT_ID
report_id=df8_N.REPORT_ID
used_credit = df8_N['USED_CREDIT_LIMIT']
credit= df8_N['CREDIT_LIMIT']
used_credit_per=used_credit/credit

####针对担保方式设置一个指标
#信用免担保的贷记卡总笔数
df3_N_free=df3_N[df3_N['guarantee_type']=='信用/免担保']
df3_N_free.head(10)
num_guarantee_free_lnd=df3_N_free.groupby('REPORT_ID').size()
#信用免担保贷款的额度
mean_guarantee_free_lnd=df3_N_free.groupby('REPORT_ID').credit_limit_amount.mean()
#训练集
df1['ovd_1_lnd']=0
df1['ovd_1_lnd']=ovd_1_lnd
df1['ovd_1_lnd']=df1['ovd_1_lnd'].fillna(0)
len(df1.ovd_1_lnd[df1.ovd_1_lnd.isnull().values==True]) #找到缺失值的位置
df1['ovd_1_lnd'].head(50)
df1['ovd_2_lnd']=0
df1['ovd_2_lnd']=ovd_2_lnd
df1['ovd_2_lnd']=df1['ovd_2_lnd'].fillna(0)
df1['ovd_2_lnd'].head(50)
df1['pay_N_per_lnd']=0
df1['pay_N_per_lnd']=pay_N
df1['pay_N_per_lnd']=df1['pay_N_per_lnd'].fillna(0)
df1['used_per']=0
df1['used_per']=used_credit_per
df1['used_per']=df1['used_per'].fillna(0)
df1['num_guarantee_free_lnd']=0
df1['num_guarantee_free_lnd']=num_guarantee_free_lnd
df1['num_guarantee_free_lnd']=df1['num_guarantee_free_lnd'].fillna(0)
df1['mean_guarantee_free_lnd']=0
df1['mean_guarantee_free_lnd']=mean_guarantee_free_lnd
df1['mean_guarantee_free_lnd']=df1['mean_guarantee_free_lnd'].fillna(0)
df1.to_csv('D:///train23.csv',index = False)

#测试集
df['ovd_1_lnd']=0
df['ovd_1_lnd']=ovd_1_lnd
df['ovd_1_lnd']=df['ovd_1_lnd'].fillna(0)
df['ovd_1_lnd'].head(50)
df['ovd_2_lnd']=0
df['ovd_2_lnd']=ovd_2_lnd
df['ovd_2_lnd']=df['ovd_2_lnd'].fillna(0)
df['ovd_2_lnd'].head(50)
df['pay_N_per_lnd']=0
df['pay_N_per_lnd']=pay_N
df['pay_N_per_lnd']=df['pay_N_per_lnd'].fillna(0)
df['used_per']=0
df['used_per']=used_credit_per
df['used_per']=df['used_per'].fillna(0)
df['num_guarantee_free_lnd']=0
df['num_guarantee_free_lnd']=num_guarantee_free_lnd
df['num_guarantee_free_lnd']=df['num_guarantee_free_lnd'].fillna(0)
df['mean_guarantee_free_lnd']=0
df['mean_guarantee_free_lnd']=mean_guarantee_free_lnd
df['mean_guarantee_free_lnd']=df['mean_guarantee_free_lnd'].fillna(0)
df.to_csv('D:///test17.csv',index = False)

####贷记卡审批成功比例
#贷记卡
df3 = pd.read_table('D:\contest_ext_crd_cd_lnd.tsv',encoding='gb2312',sep="\t")      
df3.info()
df3.index=df3.REPORT_ID
df3.head()
df3['open_date'] = pd.to_datetime(df3['open_date']) #转换为时间格式
change_df3=df3[['REPORT_ID','finance_org','currency','open_date']]
change_df3=change_df3[change_df3['currency']=='人民币'] #不包括其他币种
change_df3['open_date'] = pd.to_datetime(change_df3['open_date'])
change_df3=change_df3[(change_df3['open_date']>'2015-01-01')]
change_df3.index=change_df3.REPORT_ID
change_df3.head()
change_df3.shape #(116872, 4)
#查询记录表
df10_lnd=df10[(df10['query_reason']=='信用卡审批')]  
df10_lnd.shape   #(340989, 4)
df10_lnd.head()
#利用merge连接查询记录和贷记卡信息这两张表
df_record=pd.merge(df10_lnd,change_df3) 
df_record.head()
df_record.shape  #(1425872, 7)
df_record_1=df_record[df_record['querier']==df_record['finance_org']]
df_record_1.head()
df_record_1['open_date'] = pd.to_datetime(df_record_1['open_date'])
df_record_1['query_date'] = pd.to_datetime(df_record_1['query_date'])
df_record_2=df_record_1[df_record_1['open_date']-df_record_1['query_date']< Timedelta('30 days 00:00:00')]
df_record_3=df_record_2[df_record_2['open_date']> df_record_2['query_date']]
df_record_3.head()
df_record_3.index=df_record_3.REPORT_ID
df_duplicated = df_record_3.drop_duplicates(['REPORT_ID','open_date'],keep='last')
df_duplicated.head(5)
df_duplicated.shape  #(74195, 7)
df_duplicated=df_duplicated.drop('query_reason',axis=1)
df_duplicated.to_csv('D://train7.csv',index = False)

lnd_record=df10_lnd.groupby('REPORT_ID').size() #38206人
lnd_success=df_duplicated.groupby('REPORT_ID').size() #有29557人
df3['lnd_record']=0
df3['lnd_record']=lnd_record
len(df3.lnd_record[df3.lnd_record.isnull().values==True]) 
df3['lnd_success']=0
df3['lnd_success']=lnd_success
df3['lnd_percent']=0
df3['lnd_percent']=df3['lnd_success']/df3['lnd_record']
len(df3.lnd_percent[df3.lnd_percent.isnull().values==True])
np.mean(df3.lnd_percent)
a=0.31221359253781555
df3.lnd_percent=df3.lnd_percent.fillna(a)
lnd_percent=df3.groupby('REPORT_ID').lnd_percent.max()
#训练集
df1.index=df1.REPORT_ID
df1['recordinfo']=0
df1['recordinfo']=lnd_record
len(df1.recordinfo[df1.recordinfo.isnull().values==True]) 
df1['lnd_success']=0
df1['lnd_success']=lnd_success
df1['lnd_percent']=0
df1['lnd_percent']=lnd_percent
len(df1.lnd_percent[df1.lnd_percent.isnull().values==True])
df1['lnd_percent']=df1['lnd_percent'].fillna(0)
df1.to_csv('D://train18.csv',index = False)
#测试集
df['lnd_percent']=0
df['lnd_percent']=lnd_percent
len(df.lnd_percent[df.lnd_percent.isnull().values==True])
df['lnd_percent']=df['lnd_percent'].fillna(0)
df.to_csv('D://test12.csv',index = False)

####分类变量：学历、户籍、公积金 变成独热编码 
fea1 = pd.read_excel('D:\sel_fea.xlsx')
objectColumns = fea1.select_dtypes(include=["object"]).columns # 筛选数据类型为object的数据
fea1[objectColumns] = fea1[objectColumns].fillna("Unknown") #以分类“Unknown”填充缺失值
fea1.dtypes.value_counts() # 分类统计数据类型
fea1.isnull().sum(axis=0).sort_values(ascending=False)
dummy_df = pd.get_dummies(fea1[objectColumns])# 用get_dummies进行one hot编码
fea2 = pd.concat([fea1, dummy_df], axis=1) #当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
fea2.head()                             #查看经标准化后的数据
fea2.info()
fea2.columns
fea2.shape 
fea2=fea2.drop(objectColumns,axis=1)
#测试集
test = pd.read_excel('D:\\test.xlsx')
test.info()
test.shape
objectColumns = test.select_dtypes(include=["object"]).columns # 筛选数据类型为object的数据
test['has_fund'] = test['has_fund'].fillna("Unknown") #以分类“Unknown”填充缺失值
test[objectColumns] = test[objectColumns].fillna("Unknown") #以分类“Unknown”填充缺失值
test.dtypes.value_counts() # 分类统计数据类型
test.isnull().sum(axis=0).sort_values(ascending=False)
dummy_df = pd.get_dummies(test[objectColumns])# 用get_dummies进行one hot编码
test = pd.concat([test, dummy_df], axis=1) #当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
fea2.head()                             #查看经标准化后的数据
fea2.info()
test.columns
test.shape 
test=test.drop(objectColumns,axis=1)
test.to_csv('D://new_test.csv',index = False)












