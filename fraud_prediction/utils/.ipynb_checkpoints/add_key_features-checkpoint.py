import pandas as pd



# 数据探索，找到导致1的关键特征值
def find_key_features_info(train, selected_cols):
    features = train.drop(['Unnamed: 0', 'label'], axis=1)
    key_features_info = {}
    
    
    for selected in selected_cols:
        key_features_info[selected] = find_key_values(train, selected, 5)
    return key_features_info


def find_key_values(train, selected_feature, ratio):
    temp0 = train[train['label'] == 0]
    temp = pd.DataFrame(columns=[0,1])
    temp[0] = temp0[selected_feature].value_counts() / len(temp0)

    temp1 = train[train['label'] == 1]
    temp[1] = temp1[selected_feature].value_counts() / len(temp1)
    temp[2] = temp[1] / temp[0]
    
    #选出大于10倍的特征
    result = temp[temp[2] > ratio].sort_values(2, ascending=False).index
    
    return result


def add_key_features(train, features,  test1, selected_cols):
    
    key_features_info = find_key_features_info(train, selected_cols)
    
    for feature in key_features_info:
        if len(key_features_info[feature]) > 0:
            features[feature + '1'] = features[feature].apply(f, args=(feature, key_features_info))
            test1[feature + '1'] = test1[feature].apply(f, args=(feature, key_features_info))
    return features, test1

    
def f(x, feature, key_features_info):
    if x in key_features_info[feature]:
        return 1
    else:
        return 0
