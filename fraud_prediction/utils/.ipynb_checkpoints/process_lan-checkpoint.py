import re
import pandas as pd


def process_lan(features, test1):
    
    def f(x):
        if x in list:
            return '0'
        return x

    
    all_df = pd.concat([features, test1])
    list = all_df['lan'].value_counts()[all_df['lan'].value_counts() < 20].index
    
    
    features['lan'] = features['lan'].astype(str).apply(f)
    test1['lan'] = test1['lan'].astype(str).apply(f)

    return features, test1

    