# -*- coding: utf-8 -*-
####探索性分析
import matplotlib.pyplot as plt
plt.style.use('ggplot')  
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


####省份分布
df1 = pd.read_table('D:\contest_basic_train.tsv',encoding='gb2312',sep="\t")          #训练集
xz = pd.read_excel('D:\\xingzhengquyu.xlsx',header=0,encoding='gb2312',names=["WORK_PROVINCE","province"]) 
xz.WORK_PROVINCE=xz.WORK_PROVINCE.astype('float')
wp=df1.WORK_PROVINCE.dropna() 
wp1=df1.WORK_PROVINCE.dropna() 
df1_Y_1=df1[(df1['Y']==1)]
wp=wp.append(wp1)
wp=wp.unique()
len(wp)
wp=list(wp)
type(wp)
wp_T=xz[xz.WORK_PROVINCE.isin(wp)] #与行政区域代码进行比较
work_province_ovd = pd.merge(wp_T,df1_Y_1)  #与训练集合并
work_province = pd.merge(wp_T,df1)  #与训练集合并

##省份与违约情况的关联
both=work_province.groupby('province').size()
bad=work_province_ovd.groupby('province').size().sort_values(ascending=False)#按省份进行分组
bad_per=bad/both
bad_per=bad_per.sort_values(ascending=False)
a=bad_per.index
bad_per_df=pd.DataFrame(bad_per,columns=['percent'],index=a)
bad_per_df =bad_per_df.fillna(0)
sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})
plt.figure(figsize= (12,6))#创建画布
sns.barplot(x=a, y="percent", data=bad_per_df,ci=0)
plt.xticks(range(len(a)),a,rotation=90)
plt.show()

#对辽宁、山西、内蒙古、吉林、河北 进行编码
#训练集
work_province.to_csv('D://city_1.csv',index = False)
pd_work = pd.read_table('D://city_1.csv',encoding='gb2312',sep=",")          #训练集
pd_work.info()
pd_work.head()
pd_work.index=pd_work.REPORT_ID
province=pd_work.province
df1 = pd.read_table('D:\contest_basic_train.tsv',encoding='gb2312',sep="\t")          #训练集
df1.index=df1.REPORT_ID
df1.head()
df1['pro']=0
df1['pro']=province
df1.to_csv('D://city_3.csv',index = False)
df_work=pd.read_excel('D://city_4.xlsx',header=0,encoding='gb2312') 
df_work.info()
df_work.head()
df_work = df_work.fillna(0)
df_work.to_csv('D://city_5.csv',index = False)

#测试集
work_province2 = pd.merge(wp_T,df)  #与测试集合并
work_province2.to_csv('D://city_6.csv',index = False)
pd_work2 = pd.read_table('D://city_6.csv',encoding='gb2312',sep=",")          #训练集
pd_work2.info()
pd_work2.head()
pd_work2.index=pd_work2.REPORT_ID
province2=pd_work2.province
df = pd.read_table('D:\contest_basic_test.tsv',encoding='gb2312',sep="\t")          #训练集
df.index=df.REPORT_ID
df.head()
df['pro']=0
df['pro']=province2
df.to_csv('D://city_7.csv',index = False)
df_work3=pd.read_table('D://city_7.csv',encoding='gb2312',sep=",") 
df_work3.info()
df_work3.head()
df_work3 = df_work3.fillna(0)
df_work3.to_csv('D://city_8.csv',index = False)

##违约情况比例图
sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})
sns.set(font_scale=1)
plt.figure(figsize= (12,6))#创建画布
sns.countplot(x='Y', data=work_province)
plt.show()

X_train = pd.read_table('D:\X_train.csv',encoding='gb2312',sep=",")          
X_train.info()
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='loan_status',data=loans,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
loans['loan_status'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
plt.show()
#训练集的缺失值
df1.isnull().sum(axis=0).sort_values(ascending=False)/30000
import missingno as msno
df1_1=df1.drop(['REPORT_ID','Y'],axis=1)
df1_1.info()
msno.matrix(df1_1,figsize=(8,4))#可视化查询缺失值
objectColumns = df1.select_dtypes(include=["object"]).columns # 筛选数据类型为object的数据
df1.dtypes.value_counts() # 分类统计数据类型
