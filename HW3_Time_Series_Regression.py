#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('新竹_2019.csv', encoding='big5')
data


# In[2]:


data.drop(index = [0], axis=0, inplace=True) # row : axis = 0, column : axis = 1
data


# In[3]:


data.head(60)


# 先處理前後空白字元

# In[4]:


data.columns


# In[5]:


data.columns = data.columns.str.strip()
data.columns


# In[6]:


data.index


# In[7]:


data.loc[59, '3']


# In[8]:


columns = ['測站', '日期', '測項', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
       '22', '23']
for column in columns:
    data[column] = data[column].str.strip()


# In[9]:


data.loc[59, '3']


# 取出10.11.12月資料

# In[10]:


data.loc[:1, '日期']


# In[11]:


data.loc[data['日期'] == '2019/10/1 0:00'].index


# In[12]:


data = data.loc[data.loc[data['日期'] == '2019/10/1 0:00'].index[0] : , :]


# In[13]:


data.head(30)


# NR表示無降雨，以0取代

# In[14]:


data = data.replace(['NR'], [0.0])
data.head(30)


# 看一下有怪字元的地方

# In[15]:


data.loc[:4859, '10':]


# In[16]:


data.columns


# In[17]:


float(data.iloc[0, 13])


# 把怪怪的字元拿掉，用前後兩個值的平均

# In[18]:


symbols = ['#', '*', 'x', 'A']
for i in range(len(data)): # row
    for j in range(3, len(data.columns)): # column
        for symbol in symbols:
            if symbol in data.iloc[i, j]:
                front_j = j - 1
                back_j = j + 1
                front_i = i
                back_i = i
                if front_j == 2: # 代表取到了測項，要往前一天的最後一小時取資料
                    front_j = len(data.columns) - 1
                    front_i = i - 18 # 前一天的 row index在前面18個
                if back_j == len(data.columns): # 代表取到超過今天的數據了
                    back_j = 3 # 隔天的第一筆數據
                    back_i = i + 18 
                # 下一個的symbol不一定是上一個index的symbol，所以要再重測。
                # 因為我是用row by row, column by colum，所以不需要去測前面的
                while True:
                    s = 0 # 用來記錄是否都不是symbol
                    for symbol in symbols: 
                        if symbol in data.iloc[back_i, back_j]:
                            back_j += 1
                        else:
                            s += 1
                    if s == 4: # 代表4個symbol他都不在back裡面了
                        break
                # 因為可以讓中間都是異常值的數直接被平均數替代掉
                data.iloc[i, j:back_j] = str((float(data.iloc[front_i, front_j]) + float(data.iloc[back_i, back_j])) / 2)


# 怪字元拿掉了！！

# In[19]:


data.loc[:4859, '10':]


# 將資料切割成訓練集(10.11月)以及測試集(12月)

# In[20]:


data.loc[data['日期'] == '2019/12/31 0:00'].index[0]


# In[21]:


training_set = data.loc[data.loc[data['日期'] == '2019/10/1 0:00'].index[0] : 
                        data.loc[data['日期'] == '2019/12/1 0:00'].index[0] - 1, :]
training_set


# In[22]:


training_set.columns


# In[23]:


print(training_set.dtypes)
type(training_set)


# 再把每小時的資料從object transfer to float，testing_set也做一樣的事

# In[24]:


for column in training_set.columns[3:]:
    training_set[column] = training_set[column].astype(float)
training_set.dtypes


# In[25]:


testing_set = data.loc[data.loc[data['日期'] == '2019/12/1 0:00'].index[0]: , :]
testing_set


# In[26]:


print(testing_set.dtypes)
print(type(testing_set))


# In[27]:


for column in testing_set.columns[3:]:
    testing_set[column] = testing_set[column].astype(float)
testing_set.dtypes


# 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料
# hint: 將訓練集每18行合併，轉換成維度為(18,61*24)的DataFrame(每個屬性都有61天*24小時共1464筆資料)

# In[28]:


new_training_set = training_set.iloc[:18, :]
new_training_set


# In[29]:


# 有61筆data，i處理到 len(data) - 2 = 59就好
for i in range(int(len(training_set) / 18) - 1):
    new_training_set = pd.merge(new_training_set, training_set.iloc[(i + 1) * 18 : (i + 2) * 18, 2 : ],on='測項')
new_training_set


# In[30]:


new_training_set.drop(['測站', '日期'], axis=1, inplace=True)
new_training_set


# In[31]:


new_testing_set = testing_set.iloc[:18, :]
new_testing_set


# In[32]:


for i in range(int(len(testing_set) / 18) - 1):
    new_testing_set = pd.merge(new_testing_set, testing_set.iloc[(i + 1) * 18 : (i + 2) * 18, 2 : ],on='測項')
new_testing_set


# In[33]:


new_testing_set.drop(['測站', '日期'], axis=1, inplace=True)
new_testing_set


# 將未來第一個小時當預測目標，
# X只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值)，
# 這裡整理training_set的x, y 資料

# In[34]:


new_training_set.loc[new_training_set['測項'] == 'PM2.5'].index[0]


# In[35]:


training_x = new_training_set.iloc[new_training_set.loc[new_training_set['測項'] == 'PM2.5'].index[0], 1:]
print(training_x)
print(training_x.shape)
print(len(training_x))


# In[36]:


training_x_np = np.zeros((len(training_x) - 6, 6))
print(training_x_np)
print(training_x_np.shape[0])


# In[37]:


def get_PM25_data_x_from(data_df, data_np, amount_in_group): 
    # data_df : 原本的資料
    # data_np : 我要存的格式
    # amount_in_group : 幾個資料一組
    for i in range(len(data_df) - amount_in_group):
        k = i
        for j in range(6):
            data_np[i][j] = data_df.iloc[k, ]
            k += 1
    return data_np
training_x_np = get_PM25_data_x_from(training_x, training_x_np, 6)
print(training_x_np)
print(training_x_np[1450 : ])


# In[38]:


training_y_np = np.zeros(len(training_x_np))
def get_PM25_data_y_from(data_df, data_np, amount_in_group):
    for i in range(len(data_df) - amount_in_group):
        data_np[i] = data_df.iloc[i + amount_in_group,]
    return data_np
training_y_np = get_PM25_data_y_from(training_x, training_y_np, 6)
print(training_y_np)
print(len(training_y_np))


# 這裡整理testing_set的x, y 資料

# In[39]:


testing_x = new_testing_set.iloc[new_testing_set.loc[new_testing_set['測項'] == 'PM2.5'].index[0], 1:]
testing_x_np = np.zeros((len(testing_x) - 6, 6))
testing_x_np = get_PM25_data_x_from(testing_x, testing_x_np, 6)
print(testing_x_np)
print(len(testing_x_np))


# In[40]:


testing_y_np = np.zeros(len(testing_x_np))
testing_y_np = get_PM25_data_y_from(testing_x, testing_y_np, 6)
print(testing_y_np)
print(len(testing_y_np))


# 做Linear Regression

# '''
# The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# A constant model that always predicts the expected value of y, 
# disregarding the input features, would get a R^2 score of 0.0.
# '''

# In[41]:


def linear_regression(training_set_x, training_set_y, testing_set_x, testing_set_y):    
    x, y = training_set_x, training_set_y
    reg = LinearRegression().fit(x, y)
    predict_y = reg.predict(testing_set_x)
    y_true = testing_set_y
    y_pred = predict_y
    return mean_absolute_error(y_true, y_pred)


# '''
# 用於評估預測結果和真實資料集的接近程度的程度，其其值越小說明擬合效果越好。
# MAE output is non-negative floating point. The best value is 0.0.
# '''

# In[42]:


linear_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 做Random Forest Regression

# In[43]:


def random_forest_regression(training_set_x, training_set_y, testing_set_x, testing_set_y):    
    clf = RandomForestClassifier(max_depth=10, random_state=1)
    x, y = training_set_x, training_set_y
    y = np.array(y, dtype=int)
    clf.fit(x, y)
    predict_y = clf.predict(testing_set_x)
    y_true = testing_set_y
    y_pred = predict_y
    return mean_absolute_error(y_true, y_pred)


# In[44]:


random_forest_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 將未來第六個小時當預測目標
# X只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值)
# 這裡整理training_set的x, y 資料

# In[45]:


training_x_np = np.zeros((len(training_x) - 11, 6))
print(len(training_x_np))
training_x_np = get_PM25_data_x_from(training_x, training_x_np, 11)
print(training_x_np)


# In[46]:


training_y_np = np.zeros(len(training_x_np))
training_y_np = get_PM25_data_y_from(training_x, training_y_np, 11)
print(training_y_np)


# 這裡整理testing_set的x, y 資料

# In[47]:


testing_x = new_testing_set.iloc[new_testing_set.loc[new_testing_set['測項'] == 'PM2.5'].index[0], 1:]
testing_x_np = np.zeros((len(testing_x) - 11, 6))
testing_x_np = get_PM25_data_x_from(testing_x, testing_x_np, 11)
print(testing_x_np)
print(len(testing_x_np))


# In[48]:


testing_y_np = np.zeros(len(testing_x_np))
testing_y_np = get_PM25_data_y_from(testing_x, testing_y_np, 11)
print(testing_y_np)
print(len(testing_y_np))


# 做Linear Regression

# In[49]:


linear_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 做Random Forest Regression

# In[50]:


random_forest_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 將未來第一個小時當預測目標，
# x取所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值)，
# 先處理training set的x, y

# In[51]:


training_x = new_training_set.iloc[:, 1:]
training_x


# In[52]:


print(training_x.shape[1])


# In[53]:


training_x_np = np.zeros((training_x.shape[1] - 6, 18 * 6))
print(len(training_x_np))
print(training_x_np.shape)


# In[54]:


def get_PM25_Alldata_x_from(data_df, data_np, amount_in_group): 
    # data_df : 原本的資料
    # data_np : 我要存的格式
    # amount_in_group : 幾個資料一組
    for i in range(data_df.shape[1] - amount_in_group):
        count = 0
        for j in range(18):
            k = i
            for k in range(i, i + 6):
                data_np[i][count] = data_df.iloc[j, k]
                count += 1
                k += 1
    return data_np
training_x_np = get_PM25_Alldata_x_from(training_x, training_x_np, 6)
print(training_x_np)


# In[55]:


training_y_np = np.zeros(len(training_x_np))
def get_PM25_Alldata_y_from(data_df, data_np, amount_in_group):
    for i in range(len(data_np)):
        data_np[i] = data_df.iloc[9, i + 6]
    return data_np
training_y_np = get_PM25_Alldata_y_from(training_x, training_y_np, 6)
print(training_y_np)


# 再處理testing set的x, y

# In[56]:


testing_x = new_testing_set.iloc[:, 1:]
testing_x_np = np.zeros((testing_x.shape[1] - 6, 18 * 6))
testing_x_np = get_PM25_Alldata_x_from(testing_x, testing_x_np, 6)
print(testing_x_np)
print(len(testing_x_np))


# In[57]:


testing_y_np = np.zeros(len(testing_x_np))
print(len(testing_y_np))
testing_y_np = get_PM25_Alldata_y_from(testing_x, testing_y_np, 6)
print(testing_y_np)


# 用Linear Regression

# In[58]:


linear_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 做Random Forest Regression

# In[59]:


random_forest_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 將未來第六個小時當預測目標
# X有所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值)
# 這裡整理training_set的x, y 資料

# In[60]:


training_x = new_training_set.iloc[:, 1:]
training_x


# In[61]:


training_x_np = np.zeros((training_x.shape[1] - 11, 18 * 6))
training_x_np = get_PM25_Alldata_x_from(training_x, training_x_np, 11)
print(training_x_np)
print(len(training_x_np))


# In[62]:


training_y_np = np.zeros(len(training_x_np))
training_y_np = get_PM25_Alldata_y_from(training_x, training_y_np, 11)
print(training_y_np)
print(len(training_y_np))


# 再處理testing set的x, y

# In[63]:


testing_x = new_testing_set.iloc[:, 1:]
testing_x_np = np.zeros((testing_x.shape[1] - 11, 18 * 6))
testing_x_np = get_PM25_Alldata_x_from(testing_x, testing_x_np, 11)
print(testing_x_np)
print(len(testing_x_np))


# In[64]:


testing_y_np = np.zeros(len(testing_x_np))
testing_y_np = get_PM25_Alldata_y_from(testing_x, testing_y_np, 11) 
print(training_y_np)


# 用Linear Regression

# In[65]:


linear_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# 做Random Forest Regression

# In[66]:


random_forest_regression(training_x_np, training_y_np,testing_x_np, testing_y_np)


# In[ ]:




