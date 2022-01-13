#!/usr/bin/env python
# coding: utf-8

# ## 使用TFIDF提取文本特征
# ## 使用TFIDF + LightGBM Score=?

# In[1]:


get_ipython().run_cell_magic('time', '', "import pandas as pd\nimport pickle\nwith open('./train.pkl', 'rb') as f:\n    train = pickle.load(f)\n    \nwith open('./test.pkl', 'rb') as f:\n    test = pickle.load(f)\ntrain")


# In[2]:


# 从df种提取api_sequence
def get_apis(df):
    # 按照file_id进行分组
    group_fileid = df.groupby('file_id')
    
    # 统计file_id 和对应的 api_sequence
    file_api = {}
    
    # 计算每个file_id的api_sequence
    for file_id, file_group in group_fileid:
        # 针对file_id 按照线程tid 和 顺序index进行排序
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        # 得到api的调用序列
        api_sequence = ' '.join(result['api'])
        #print(api_sequence)
        file_api[file_id] = api_sequence
    return file_api

train_apis = get_apis(train)
len(train_apis)


# In[9]:


get_ipython().run_cell_magic('time', '', 'test_apis = get_apis(test)\nlen(test_apis)')


# In[12]:


import pickle
with open('./df_train.pkl', 'rb') as file:
    df_train = pickle.load(file)
    
with open('./df_test.pkl', 'rb') as file:
    df_test = pickle.load(file)
df_train


# In[14]:


df_train.drop(['api', 'tid', 'index'], axis=1, inplace=True)
df_test.drop(['api', 'tid', 'index'], axis=1, inplace=True)


# In[16]:


temp = pd.DataFrame.from_dict(train_apis, orient='index', columns=['api'])
temp = temp.reset_index().rename(columns={'index': 'file_id'})
df_train = df_train.merge(temp, on='file_id', how='left')
df_train


# In[17]:


temp = pd.DataFrame.from_dict(test_apis, orient='index', columns=['api'])
temp = temp.reset_index().rename(columns={'index': 'file_id'})
df_test = df_test.merge(temp, on='file_id', how='left')
df_test


# In[21]:


df_all = pd.concat([df_train, df_test], axis=0)
df_all


# In[23]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import TfidfVectorizer\n# 使用1-3元语法（1元语法 + 2元语法 + 3元语法）\n# 将min_df = 0.1 => TFIDF特征数大幅减少 => 速度快\n# 也可以设置 max_df = 0.8\n# 也可以用 PCA降维\nvec = TfidfVectorizer(ngram_range=(1, 3), min_df=0.01)\napi_features = vec.fit_transform(df_all['api'])\napi_features")


# In[26]:


df_apis = pd.DataFrame(api_features.toarray(), columns=vec.get_feature_names())
df_apis.to_pickle('./df_apis.pkl')
df_apis


# In[30]:


#df_train_apis = 
df_train_apis = df_apis[df_apis.index <= 13886]
df_test_apis = df_apis[df_apis.index > 13886]
df_test_apis


# In[31]:


df_test_apis.index = range(len(df_test_apis))
df_test_apis


# In[32]:


# 将tfidf特征 与原特征进行合并
df_train = df_train.merge(df_train_apis, left_index=True, right_index=True)
df_test = df_test.merge(df_test_apis, left_index=True, right_index=True)
df_train


# In[46]:


df_train.to_pickle('./df_train2.pkl')
df_test.to_pickle('./df_test2.pkl')


# In[34]:


# 查看某个变量的资源使用
import sys
sys.getsizeof(df_train) / 1024 /1024 # M


# In[45]:


#print(df_train.dtypes.values)
#df_train.select_dtypes(include='O')
df_train.drop('api', axis=1, inplace=True)
df_test.drop('api', axis=1, inplace=True)


# In[47]:


get_ipython().run_cell_magic('time', '', "import lightgbm as lgb\nclf = lgb.LGBMClassifier(\n            num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='multiclass',\n            max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2021,\n            n_estimators=2000, subsample=1, colsample_bytree=1)\nclf.fit(df_train.drop('label', axis=1), df_train['label'])")


# In[26]:


result = clf.predict_proba(df_test)
result
result_lgb = pd.DataFrame(result, columns=['prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7'])
result_lgb['file_id'] = df_test['file_id'].values
result_lgb


# In[ ]:


columns = ['file_id', 'prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7']
result_xgb.to_csv('./baseline_lgb_2000_tfidf.csv', index=False, columns=columns)


# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBClassifier(
            max_depth=9, learning_rate=0.005, n_estimators=2000, 
            objective='multi:softprob', tree_method='gpu_hist', 
            subsample=0.8, colsample_bytree=0.8, 
            min_child_samples=3, eval_metric='logloss', reg_lambda=0.5)
model_xgb.fit(df_train.drop('label', axis=1), df_train['label'])


# In[ ]:


result_xgb = model_xgb.predict_proba(df_test)

result_xgb = pd.DataFrame(result_xgb, columns=['prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7'])
result_xgb['file_id'] = df_test['file_id'].values
result_xgb


# In[29]:


# 对两个模型的结果 进行加权平均
result = result_lgb.copy()
weight_lgb, weight_xgb = 0.5, 0.5
result['prob0'] = result['prob0'] * weight_lgb + result_xgb['prob0'] * weight_xgb
result['prob1'] = result['prob1'] * weight_lgb + result_xgb['prob1'] * weight_xgb
result['prob2'] = result['prob2'] * weight_lgb + result_xgb['prob2'] * weight_xgb
result['prob3'] = result['prob3'] * weight_lgb + result_xgb['prob3'] * weight_xgb
result['prob4'] = result['prob4'] * weight_lgb + result_xgb['prob4'] * weight_xgb
result['prob5'] = result['prob5'] * weight_lgb + result_xgb['prob5'] * weight_xgb
result['prob6'] = result['prob6'] * weight_lgb + result_xgb['prob6'] * weight_xgb
result['prob7'] = result['prob7'] * weight_lgb + result_xgb['prob7'] * weight_xgb

columns = ['file_id', 'prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7']
result.to_csv('./baseline_lgb_xgb_2000_tfidf.csv', index=False, columns=columns)


# In[30]:


columns = ['file_id', 'prob0','prob1','prob2','prob3','prob4','prob5','prob6','prob7']
result.to_csv('./baseline_lgb_2000.csv', index=False, columns=columns)


# In[31]:


result = pd.read_csv('./baseline_lgb_2000.csv')
result

