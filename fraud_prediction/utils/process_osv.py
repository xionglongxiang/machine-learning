import re


def process_osv(features, test1):
    
    features['osv'] = features['osv'].apply(f)
    test1['osv'] = test1['osv'].apply(f)

    return features, test1


def f(x):

    x = str(x)
    x = x.replace('Android_', '')
    x = x.replace('Android ', '')
    
    if x == '%E6%B1%9F%E7%81%B5OS+5.0':
        return 5.0
    if x == '6.0 十核2.0G_HD':
        return 6.0
    if x == 'f073b_changxiang_v01_b1b8_20180915':
        return 0
    if x == '6.0.1_19':
        return 6.01
    if x == 'nan':
        return 0
    if x == 'GIONEE_YNGA':
        return 0
    
    if x == '4.4W':
        return 4.4
    
    if x.find('.') > 0:
        index = x.find('.')    
        x = x[0:index] + '.' + re.sub(r'\D', "", x[index+1:])[0:2]
    x = float(x)
    
        
        
    if x > 10000:
        return x / 10000
    if x > 1000:
        return x / 1000
    
    return x

