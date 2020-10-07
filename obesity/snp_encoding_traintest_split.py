import pandas as pd 
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
import json

# generate integer encoding table(dict)
def integer_encode_table():
    snp_uni = ['00', 'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC','GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    integer = list(range(len(snp_uni)))
    snp_uni
    table = {}
    for index,key in enumerate(snp_uni):
        table[key] = integer[index]
    return table

# generate SNP encoding table(dict)
def SNP_encode_table():
    table=integer_encode_table()
    snp_table={}
    x=[]
    for key,value in table.items():
        if value == 0:
            x.append(value)
            snp_table[key] = value
            continue
        else:
            for other_key in list(snp_table.keys()):
                if set(key) == set(other_key): 
                    snp_table[key]=snp_table[other_key]
                    value = snp_table[other_key]
                continue
            x.append(value)
            while value-1 not in x:
                x[-1] = value-1
                value = value -1       
            snp_table[key]=value
    return snp_table

# randomly sample data from normal SNP data to make the amount is equal to obesity SNP data
def random_sample_normal_data_equal_to_obesity(obesity_snp_data_path, normal_snp_data_path):
    # load data
    O_data = pd.read_table(obesity_snp_data_path, sep= '\t', low_memory=False)
    N_data = pd.read_table(normal_snp_data_path, sep= '\t', low_memory=False)
    # random sample 
    random.seed(10) 
    N_data_sampled = N_data.loc[random.sample(list(N_data.index.values),len(O_data))]
    return O_data, N_data_sampled

# concatenate obesity and normal SNP data
# give labels
# normal : 0 , obesity : 1
def snp_encoding_and_labeling(O_data, N_data_sampled):
    # concatenate data
    data = pd.concat([N_data_sampled,O_data])
    # SNP_encoding then convert data format to np.array
    data_encoded = np.array(data.replace(SNP_encode_table()))
    # give labels 
    labels = np.hstack((np.repeat(0, len(N_data_sampled)),np.repeat(1, len(O_data))))
    return data_encoded, labels

# train test split
# '/home/obesity/json_data/TWB2_male_3060_bmi2430_exclude_random7500_1_ped'
def split(data_encoded, labels, test_size=0.1, random_state = 10, save = True, output_root = None):
    train_data, test_data, train_labels, test_labels = train_test_split(data_encoded, labels, test_size=0.1, random_state = 10, stratify = labels)
    if save:
        output_root = output_root or os.getcwd()
        if not os.path.isdir(output_root):
            os.mkdir(output_root)
        json.dump(train_data.tolist(), open(os.path.join(output_root,'train_data.json'), "w"), indent=4) 
        json.dump(test_data.tolist(), open(os.path.join(output_root,'test_data.json'), "w"), indent=4) 
        json.dump(train_labels.tolist(), open(os.path.join(output_root,'train_labels.json'), "w"), indent=4) 
        json.dump(test_labels.tolist(), open(os.path.join(output_root,'test_labels.json'), "w"), indent=4) 
        print('train test encoded data and labels saved')
    return  train_data, test_data, train_labels, test_labels

# get training data and labels for 10 cross validation
def get_balanced_encoded_train_data_and_labels(obesity_snp_data_path, normal_snp_data_path):
    O_data, N_data_sampled = random_sample_normal_data_equal_to_obesity(obesity_snp_data_path, normal_snp_data_path)
    data_encoded, labels = snp_encoding_and_labeling(O_data, N_data_sampled)
    train_data, test_data, train_labels, test_labels = split(data_encoded, labels, test_size=0.1, random_state = 10, save=False)
    return train_data, train_labels