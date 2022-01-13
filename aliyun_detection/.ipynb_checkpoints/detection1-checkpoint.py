#!/usr/bin/env python
# coding: utf-8

# ## 使用file_id的统计特征：count, nunique, min, max, mean, median, std, ptp
# ## score = 0.733902

# In[33]:


import pandas as pd

# 数据加载
def get_data(file_name):
    result = []
    chunk_index = 0
    for df in pd.read_csv(open(file_name, 'r'), chunksize = 1000000):
        result.append(df)
        print('chunk', chunk_index)
        chunk_index += 1
    result = pd.concat(result, ignore_index=True, axis=0)
    return result

# 获取全量数据
train = get_data('./security_train.csv')
train


# In[4]:


# 获取全量数据
test = get_data('./security_test.csv')
test


# In[3]:


# 13887个file_id
train['file_id'].value_counts()


# In[1]:


import os
import psutil
mem = psutil.virtual_memory()
print('总内存：',mem.total/1024/1024)
print('已使用内存：', mem.used/1024/1024)
print('空闲内存：', mem.free/1024/1024)
print('使用占比：',mem.percent)
print('当前线程PID：', os.getpid())


# In[34]:


"""
# 文件以python对象格式进行保存
import pickle
with open('./train.pkl', 'wb') as f:
    pickle.dump(train, f)

with open('./test.pkl', 'wb') as f:
    pickle.dump(test, f)
"""


# In[35]:


get_ipython().run_cell_magic('time', '', "import pickle\nwith open('./train.pkl', 'rb') as f:\n    train = pickle.load(f)\n    \nwith open('./test.pkl', 'rb') as f:\n    test = pickle.load(f)")


# In[6]:


import pandas as pd
# 对api字段进行LabelEncoder
#train['api'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# 将训练集 和 测试集进行合并
df_all = pd.concat([train, test])
df_all['api'] = le.fit_transform(df_all['api'])
df_all[['api']]


# In[8]:


train


# In[9]:


# 提取train['api']
train['api'] = df_all[df_all['label'].notnull()]['api']
test['api'] = df_all[df_all['label'].isnull()]['api']
train


# In[14]:


"""
# 查看某个变量的资源使用
import sys
sys.getsizeof(df_all) / 1024 /1024 # M
"""


# In[12]:


# 针对不用的变量，进行内存释放
import gc
del df_all
gc.collect()


# In[20]:


get_ipython().run_cell_magic('time', '', "# 构造新的特征（基于file_id的聚合统计）\ndef get_features(df):\n    # 按照file_id分组，提取统计特征\n    df_file = df.groupby('file_id')\n    # df1为最终的结果\n    if 'label' in df.columns: # 训练集\n        df1 = df.drop_duplicates(subset=['file_id', 'label'], keep='first')\n    else: # 测试集\n        df1 = df.drop_duplicates(subset=['file_id'], keep='first')\n    df1 = df1.sort_values('file_id')\n    # 提取多个特征的 统计特征 api, tid, index\n    features = ['api', 'tid', 'index']\n    for f in features:\n        # 针对file_id 构造不同特征， 一个file_id 只有一行数据\n        df1[f+'_count'] = df_file[f].count().values\n        df1[f+'_nunique'] = df_file[f].nunique().values\n        df1[f+'_min'] = df_file[f].min().values\n        df1[f+'_max'] = df_file[f].max().values\n        df1[f+'_mean'] = df_file[f].mean().values\n        df1[f+'_median'] = df_file[f].median().values\n        df1[f+'_std'] = df_file[f].std().values\n        df1[f+'_ptp'] = df1[f+'_max'] - df1[f+'_min']\n    return df1\n\ndf_train = get_features(train)\ndf_train")


# In[ ]:


# XGBoost 有GPU的版本


# In[21]:


get_ipython().run_cell_magic('time', '', 'df_test = get_features(test)\ndf_test')


# In[22]:


df_train.to_pickle('./df_train.pkl')
df_test.to_pickle('./df_test.pkl')


# In[36]:


import pickle
with open('./df_train.pkl', 'rb') as file:
    df_train = pickle.load(file)

with open('./df_test.pkl', 'rb') as file:
    df_test = pickle.load(file)


# In[37]:


get_ipython().run_cell_magic('time', '', "import lightgbm as lgb\nclf = lgb.LGBMClassifier(\n            num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='multiclass',\n            max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2021,\n            n_estimators=2000, subsample=1, colsample_bytree=1)\nclf.fit(df_train.drop(['file_id','label'], axis=1), df_train['label'])")


# In[39]:


# ' '.join 用 ' '做拼接间隔符
result = clf.predict_proba(df_test.drop('file_id', axis=1))
result


# In[40]:


result = pd.DataFrame(result, columns=['prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7'])
result['file_id'] = df_test['file_id'].values
result


# In[41]:


columns = ['file_id', 'prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7']
result.to_csv('./baseline_lgb_2000_file_id.csv', index=False, columns=columns)


# In[42]:


result = pd.read_csv('./baseline_lgb_2000_file_id.csv')
result

