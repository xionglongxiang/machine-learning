import re
import pandas as pd


def process_android_id(features, test1):
    
    def f(x):
        if x not in list:
            return 10000
        return x

    
    all_df = pd.concat([features, test1])
    list = all_df['android_id'].value_counts()[all_df['android_id'].value_counts() > 50].index
    
    print('len(list)', list)
    
    
    
    features['android_id'] = features['android_id'].astype(str).apply(f)
    test1['android_id'] = test1['android_id'].astype(str).apply(f)

    return features, test1

