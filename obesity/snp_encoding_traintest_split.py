import pandas as pd 
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import json
from io import StringIO

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

# replacer = Replacer(table)
# SNP_data.replace(SNP_encode_table())
class Replacer:
    def __init__(self, table):
        self._table = table
        
    @property
    def table(self):
        return self._table
    
    def __call__(self, x):
        if isinstance(x,str):
            x = self._table[x]
        else:
            x = x.apply(self)
        return x

# read_obesity_and_normal_snp_data
def read_obesity_and_normal_snp_data(obesity_snp_data_path, normal_snp_data_path):
    # load data
    O_data = pd.read_table(obesity_snp_data_path, sep= '\t', low_memory=False)
    N_data = pd.read_table(normal_snp_data_path, sep= '\t', low_memory=False)
    return O_data, N_data

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



# get testing data and labels used in 10 cross validation
def get_balanced_encoded_test_data_and_labels(obesity_snp_data_path, normal_snp_data_path):
    O_data, N_data_sampled = random_sample_normal_data_equal_to_obesity(obesity_snp_data_path, normal_snp_data_path)
    data_encoded, labels = snp_encoding_and_labeling(O_data, N_data_sampled)
    train_data, test_data, train_labels, test_labels = split(data_encoded, labels, test_size=0.1, random_state = 10, save=False)
    return test_data, test_labels



# fast encoding (updated on 20210503)
def fast_encoding(snp_data_path):
    with open(snp_data_path,'r') as fp:
        snp_data_string = fp.read()
    rows = snp_data_string.split('\n')
    head = rows[0]
    body = '\n'.join(rows[1:])
    for key,value in SNP_encode_table().items():
        body = body.replace(key,str(value))

    new_data = StringIO(head+"\n"+body)
    #with open("temp.tsv",'w') as fp:
    #    fp.write(new_data)
    df = pd.read_csv(new_data, sep="\t")
    return df

# input snp data path
# concatenate obesity and normal SNP data
# give labels
# normal : 0 , obesity : 1
def snp_fast_encoding_and_labeling(normal_snp_data_path, obesity_snp_data_path):
    # data encoding
    encoded_N_data = fast_encoding(normal_snp_data_path)
    encoded_O_data = fast_encoding(obesity_snp_data_path)
    # concatenate encoded data
    data_encoded = pd.concat([encoded_N_data, encoded_O_data])
    # give labels 
    labels = np.hstack((np.repeat(0, len(encoded_N_data)),np.repeat(1, len(encoded_O_data))))
    return data_encoded, labels

def snp_fast_encoding_and_labeling_p(normal_snp_data_path, obesity_snp_data_path,
                                   normal_phenotype_path=None, obesity_phenotype_path=None, balance=False, phenotype=True):
    # data encoding
    encoded_N_data = fast_encoding(normal_snp_data_path)
    encoded_O_data = fast_encoding(obesity_snp_data_path)
    
    # random sample
    if balance:
        random.seed(10) 
        encoded_N_data = encoded_N_data.loc[random.sample(list(encoded_N_data.index.values),len(encoded_O_data))]
    # concatenate encoded data
    data_encoded = pd.concat([encoded_N_data, encoded_O_data], ignore_index=True)
    # give labels 
    labels = np.hstack((np.repeat(0, len(encoded_N_data)),np.repeat(1, len(encoded_O_data))))
    
    # scale phenotype data #, 'SEX', 'MESS_CURR'
    if phenotype:
        normal_phenotype = pd.read_csv(normal_phenotype_path, index_col=0)
        obesity_phenotype = pd.read_csv(obesity_phenotype_path, index_col=0)
        # filter phenotype data
        normal_phenotype = normal_phenotype.iloc[list(encoded_N_data.index.values)][['DIABETES_SELF', 'AGE']]
        obesity_phenotype = obesity_phenotype[['DIABETES_SELF', 'AGE']]
        print('normal:{}\nobesity:{}'.format(len(normal_phenotype), len(obesity_phenotype)))
        # concat normal and obesity phenotype data
        data_phenotype = pd.concat([normal_phenotype, obesity_phenotype],ignore_index=True)
        # normalize min:-1 max:1 
        Min_Max_Scaler = MinMaxScaler(feature_range=(-1,1)) 
        scaled_data_phenotype = Min_Max_Scaler.fit_transform(data_phenotype)
        return data_encoded, labels, scaled_data_phenotype
    else:
        return data_encoded, labels

# scale phenotype data
def get_scaled_phenotype_data(normal_phenotype_path, obesity_phenotype_path):
    # read phenotype
    normal_phenotype = pd.read_csv(normal_phenotype_path, index_col=0)
    obesity_phenotype = pd.read_csv(obesity_phenotype_path, index_col=0)
    # filter phenotype data
    normal_phenotype = normal_phenotype[['DIABETES_SELF', 'AGE', 'SEX', 'MESS_CURR']]
    obesity_phenotype = obesity_phenotype[['DIABETES_SELF', 'AGE', 'SEX', 'MESS_CURR']]
    print('normal:{}\nobesity:{}'.format(len(normal_phenotype), len(obesity_phenotype)))
    # concat normal and obesity phenotype data
    data_phenotype = pd.concat([normal_phenotype, obesity_phenotype], sort=False)
    # normalize min:-1 max:1 
    Min_Max_Scaler = MinMaxScaler(feature_range=(-1,1)) 
    scaled_data_phenotype = Min_Max_Scaler.fit_transform(data_phenotype)
    return scaled_data_phenotype