import config

import glob
import random
random.seed(1)

def read_files(file_list):
    texts=[]
    for f_name in file_list:
        with open(f_name) as f:
            txt=f.read().strip()
            texts.append(txt)
    return texts

def read_data(section):
    """section is train or test"""
    data_pos=read_files(sorted(glob.glob("aclImdb/{}/pos/*.txt".format(section)))[:10000])
    data_neg=read_files(sorted(glob.glob("aclImdb/{}/neg/*.txt".format(section)))[:10000])
    classes=["pos"]*len(data_pos)+["neg"]*len(data_neg)
    data=data_pos+data_neg
    assert len(classes)==len(data)
    data_classes=list(zip(data,classes))
    random.shuffle(data_classes)
    return list(x[0] for x in data_classes),list(x[1] for x in data_classes)


    
    
