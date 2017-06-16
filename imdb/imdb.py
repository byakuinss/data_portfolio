### 資料擷取與匯入 ###
import pandas as pd
df=pd.read_csv("movie_metadata_my.csv", encoding='big5')

### 資料預處理 ###

# 檢查遺漏值
df.isnull()
# 去除有遺漏值的row
df=df.dropna()
# 更改欄位型態
df['評論'] = df['評論'].astype(int)
df['票房'] = df['票房'].astype(int)
df['IMDB用戶投票數量'] = df['IMDB用戶投票數量'].astype(int)
df['預算'] = df['預算'].astype(int)
df['上線日期'] = df['上線日期'].astype(int)
df['IMDB評分'] = df['IMDB評分'].astype(int)
df['電影FB粉絲數'] = df['電影FB粉絲數'].astype(int)
df['導演名稱'] = df['導演名稱'].astype(str)
df['電影名稱'] = df['電影名稱'].astype(str)
df['語言'] = df['語言'].astype(str)
df['國家'] = df['國家'].astype(str)
#IMDB評分分組轉換
bins=[0,2,4,6,8,10]
labels=['E','D','C','B','A']
df['IMDB評分分組']=pd.cut(df['IMDB評分'],bins,right=False,labels=labels)

# 資料聚合: 導演+電影組合的IMDB評分平均值
df_group_mean=df[['導演名稱','IMDB評分','電影名稱']].groupby(['導演名稱','電影名稱']).mean()
# 取出評分最高分前10名的導演+電影組合
df_group_mean.sort_values(by='IMDB評分',ascending=[0]).head(10)

# 樞鈕分析:每年IMDB評分分組分佈狀況
df_pvt=df.pivot_table(values='電影名稱', index=['上線日期'], columns='IMDB評分分組', aggfunc='count') 
df_pvt=df_pvt.fillna(0)

# 樞鈕分析:不同導演的電影IMDB評分分組分佈狀況
df_pvt2=df.pivot_table(values='電影名稱', index=['導演名稱'], columns='IMDB評分分組', aggfunc='count') 
df_pvt2=df_pvt2.fillna(0)


### 資料分析 ###

# 統計分析
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('ggplot')

%matplotlib inline

df.describe()

# 相關性分析
plt.rcParams['font.family']='DFKai-SB'
df.corr()

# High Correlation
df.plot(kind='scatter',title='評論 ~ 電影FB粉絲數: 高度相關',figsize=(6,4),x='評論',y='電影FB粉絲數',marker='+')
df.plot(kind='scatter',title='票房 ~ IMDB用戶投票數量: 高度相關',figsize=(6,4),x='票房',y='IMDB用戶投票數量',marker='+')
df.plot(kind='scatter',title='評論 ~ IMDB用戶投票數量: 中度相關',figsize=(6,4),x='評論',y='IMDB用戶投票數量',marker='+')


### 視覺化 ###

# 折線圖
df[['電影FB粉絲數','評論','IMDB用戶投票數量']].plot(kind='line',title='line chart',figsize=(8,6),subplots=True,style={'電影FB粉絲數:':'-','評論':'--','IMDB用戶投票數量':'-.'})

# 機率密度圖
df['電影FB粉絲數'].plot(kind='kde',title='kde of movie_facebook_likes',figsize=(6,4))

plt.rcParams['axes.unicode_minus']=False # 正常顯示負號
df.plot(kind='hexbin',title='hexbin of IMDB_score & gross', x='票房',y='IMDB評分',gridsize=10)
# 頻率越高，顏色越深

# 散佈圖
df[['IMDB用戶投票數量','IMDB評分','預算']].plot(kind='scatter',title='Scatter',figsize=(6,4),x='IMDB用戶投票數量',y='預算',c='IMDB評分')


### 基礎機器學習 ###

# 線性迴歸與評估

# 去除極端值
df = df[df['電影FB粉絲數']<250000]
df = df[df['評論']<800]

# 重畫散佈圖
df.plot(kind='scatter',title='Reviews ~ Gross: High Correlation',figsize=(6,4),x='評論',y='電影FB粉絲數',marker='+')

# 切分資料集
from sklearn.cross_validation import train_test_split
x=df[['電影FB粉絲數']]
y=df[['評論']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) # 30% for testing, 70% for training
x_train.head()

# 開始訓練簡單迴歸
from sklearn import datasets, linear_model
plt.style.use('ggplot')

# 建立 linea Regression 物件
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(x_train, y_train)

print('各變項參數: \n', regr.coef_)
print('模型截距：', regr.intercept_)

print("均方誤差 (MSE): %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))

# 模型在测试集上的得分，得分结果在0到1之间，数值越大，说明模型越好
print('模型得分: %.2f' % regr.score(x_test, y_test))

plt.scatter(x_test, y_test, color='blue', marker='x')
plt.plot(x_test, regr.predict(x_test), color='green', linewidth=1)

plt.ylabel('評論')
plt.xlabel('電影FB粉絲數')

plt.show()


# 切分資料集
x=df[['IMDB用戶投票數量']]
y=df[['評論']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) # 30% for testing, 70% for training
x_train.head()

# 開始訓練簡單迴歸
plt.style.use('ggplot')

# 建立 linea Regression 物件
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(x_train, y_train)

print('各變項參數: \n', regr.coef_)
print('模型截距：', regr.intercept_)

print("均方誤差 (MSE): %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))

# 模型在测试集上的得分，得分结果在0到1之间，数值越大，说明模型越好
print('模型得分: %.2f' % regr.score(x_test, y_test))

plt.scatter(x_test, y_test, color='blue', marker='x')
plt.plot(x_test, regr.predict(x_test), color='green', linewidth=1)

plt.ylabel('評論')
plt.xlabel('IMDB用戶投票數量')

plt.show()


# 資料集切分 (2 feature)
x=df[['IMDB用戶投票數量', '電影FB粉絲數']]
y=df[['評論']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) # 30% for testing, 70% for training
x_train.head()

# 標準化
sc = StandardScaler()
sc.fit(x_train)
x_train_nor = sc.transform(x_train)
x_test_nor = sc.transform(x_test)
x_train_nor[:10]

# 開始訓練
plt.style.use('ggplot')

# 建立 linea Regression 物件
regr = linear_model.LinearRegression()

# 訓練模型
regr.fit(x_train_nor, y_train)

print('各變項參數: \n', regr.coef_) 
print("均方誤差 (MSE): %.2f" % np.mean((regr.predict(x_test_nor) - y_test) ** 2))
print('模型截距：', regr.intercept_)

# 模型在测试集上的得分，得分结果在0到1之间，数值越大，说明模型越好
print('模型得分: %.2f' % regr.score(x_test_nor, y_test))

