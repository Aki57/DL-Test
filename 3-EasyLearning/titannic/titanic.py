#%%
"""
1.处理数据
"""
import pandas as pd
import re

def get_title(name):
    if(pd.isnull(name)):
        return 'Null'

    title_search = re.search('([A-Za-z]+)\.', name)
    
    if title_search:
        return title_search.group(1).lower()
    else:
        return 'None'
    
titles = {'mr': 1,
          'mrs': 2, 'mme': 2,
          'ms': 3, 'miss': 3, 'mlle': 3,
          'don': 4, 'sir': 4, 'jonkheer': 4,
          'major': 4, 'col': 4, 'dr': 4, 'master': 4, 'capt': 4,
          'dona': 5, 'lady': 5, 'countness': 5,
          'rev': 6}

def get_train_data(path):
    # 从CSV文件中读入数据
    data = pd.read_csv(path)
    
    # 取部分特征字段用于分类，清洗Age等数据
    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    mean_age = data["Age"].mean()
    data.loc[data.Age.isnull(), "Age"] = mean_age
    
    data['Title'] = data['Name'].apply(lambda name: titles.get(get_title(name)))
    data['Honor'] = data['Title'].apply(lambda title: 1 if title == 4 or title == 5 else 0)
    
    data['Embarked'] = data['Embarked'].apply(lambda embarked: 1 if embarked == 'S' else embarked)
    data['Embarked'] = data['Embarked'].apply(lambda embarked: 2 if embarked == 'C' else embarked)
    data['Embarked'] = data['Embarked'].apply(lambda embarked: 3 if embarked == 'Q' else embarked)
    
    # 两种分类分别是幸存和死亡，‘Survived’字段是其中一种分类的标签，
    # 新增‘Deceased’字段表示第二段分类的标签，取值为‘Survived’字段取非
    data['Deceased'] = data['Survived'].apply(lambda s: int(not s))
    
    data = data.fillna(0)
    data.info()
    
    return data

def get_test_data(path):
    # 从CSV文件中读入数据
    data = pd.read_csv(path)
    
    # 取部分特征字段用于分类，清洗Age等数据
    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    mean_age = data["Age"].mean()
    data.loc[data.Age.isnull(), "Age"] = mean_age
    
    data['Title'] = data['Name'].apply(lambda name: titles.get(get_title(name)))
    data['Honor'] = data['Title'].apply(lambda title: 1 if title == 4 or title == 5 else 0)
    
    data['Embarked'] = data['Embarked'].apply(lambda embarked: 1 if embarked == 'S' else embarked)
    data['Embarked'] = data['Embarked'].apply(lambda embarked: 2 if embarked == 'C' else embarked)
    data['Embarked'] = data['Embarked'].apply(lambda embarked: 3 if embarked == 'Q' else embarked)
    
    data = data.fillna(0)
    data.info()
    
    return data

#%%
data = get_train_data('../../dataset/titanic/train.csv')

dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Honor', 'Embarked']].as_matrix()
dataset_Y = data[['Survived']].as_matrix()

#%%
"""
2.初步处理数据
"""
from sklearn.model_selection import train_test_split

# 使用sklearn的train_test_split函数将标记数据切分为“训练数据集和验证集”
# 将全部标记数据随机洗牌后切分，其中验证数据占20%，由test_size参数指定
X_train, X_test, y_train, y_test = train_test_split(
    dataset_X, dataset_Y, test_size = 0.2, random_state = 42)

#%%
"""
3.使用SkFlow建立模型
"""
import tensorflow.contrib.learn as skflow
from sklearn import metrics

feature_cols = skflow.infer_real_valued_columns_from_input(X_train)
classifier = skflow.LinearClassifier(feature_columns=feature_cols, n_classes=2)
classifier.fit(X_train, y_train, steps=500)

#%%
# 预测结果有些问题
y_pred = classifier.predict(X_test)
print(y_pred)

#%%
accuracy = metrics.accuracy_score(y_test, y_test)
print("Accuracy: %f" % accuracy)

#%%
"""
4.TFLearn
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn

train_data = get_train_data('../../dataset/titanic/train.csv')
X = train_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Honor']].as_matrix()
Y = train_data[['Deceased', 'Survived']].as_matrix()

# mkdir checkpoint path
ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
# 定义分类模型
x_input = tflearn.input_data([None, X.shape[1]])
y_pred = tflearn.layers.fully_connected(x_input, 2, activation='softmax')
net = tflearn.regression(y_pred)
model = tflearn.DNN(net)

# 读取模型存档
# if os.path.isfile(os.path.join(ckpt_dir, 'model.cpkt')):
#     model.load(os.path.join(ckpt_dir, 'model.ckpt'))
# 训练    
model.fit(X, Y, validation_set=0.1, n_epoch=50)
# 存储模型参数
model.save(os.path.join(ckpt_dir, 'model.ckpt'))
# 查看模型在训练集上的准确度
metric = model.evaluate(X, Y)
print('Accuracy on train set: %.9f' % metric[0])

#%%
# 读取测试数据，并进行预测
test_data = get_test_data('../../dataset/titanic/test.csv')
X = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Honor']].as_matrix()
predictions = np.argmax(model.predict(X), axis=1)

# 构建提交结果的数据结构，并将结果存为csv文件
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
submission.to_csv("../../dataset/titanic/titanic_submission.csv", index=False)
