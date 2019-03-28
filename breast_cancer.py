# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:56:39 2019

@author: hardyliu
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#加载数据集
data = pd.read_csv('data.csv')

#初步探索数据，由于字段比较多，设置显示全部字段
pd.set_option('display.max_columns',None)
print(data.columns)
print(data.head())
print(data.describe())

#将特征字段分成三组
features_mean =list(data.columns[2:12])
features_se=list(data.columns[12:22])
features_worst = list(data.columns[22:32])

#进一步的数据清洗，删除没有用的ID列
data.drop("id",axis=1,inplace=True)
#标签列定义，防止手误
target_label = 'diagnosis'
#将该列值的字符映射为0，1
data[target_label]=data[target_label].map({'M':1,'B':0})

#将诊断结果可视化
sns.countplot(data[target_label],label='Count')
plt.show()

#用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
#annot=True显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()


# 经过相关性分析比较后，对数据进行进一步的降维处理，选择以下特征作为后续模型需要处理的列
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 

#抽取30%的数据作为测试集
train,test= train_test_split(data,test_size=0.3)

train_x = train[features_remain]
train_y = train[target_label]

test_x = test[features_remain]
test_y = test[target_label]

#对数据进行规范化处理，让数据尽量保持在同一量级上，避免因为维度问题造成的误差
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)


#创建SVM分类器
model = SVC()
#用训练集做训练
model.fit(train_x,train_y)
#用测试集做预测
prediction = model.predict(test_x)

print('准确率评估：',accuracy_score(prediction,test_y))
