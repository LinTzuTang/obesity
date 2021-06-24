import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# A A to AA
def ATCG_table():
    table = {}
    for first in list('ATCG'):
        for second in list('ATCG'):
            table[first+" "+second] = first+second
    table['0 0'] = '00'
    return table

#replacer = Replacer(table)
class Replacer:
    def __init__(self, table):
        self._table = table
        
    @property
    def table(self):
        return self._table
    
    def __call__(self, x):
        x = self._table[x]
        return x

# write train & validation snp data    
# ped_path = '/home/obesity/input_data/ped_1112/1105male_train_qc_clump_ped.ped'
# output_root='/home/obesity/snp_data/snp_data_1112' 
def write_train_validation_snp_data(ped_path, output_root, gender='Male'):
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    # read ped file
    ped = pd.read_table(ped_path, sep= '\t', low_memory=False, header= None)
    # pre-save id
    id_name = ped[0]
    # remain just snp data
    ped = ped.iloc[:,6:]
    ped.columns = ["SNP_{}".format(i) for i in range(1,len(ped.columns)+1)]
    # replace A A to AA
    replacer = Replacer(ATCG_table())
    for index in range(len(ped.columns)):
        ped.iloc[:,index] = ped.iloc[:,index].apply(replacer)
    # re-give id    
    ped.insert(loc=0, column='id', value=id_name)
    # train validation data split
    train, valid = train_test_split(ped, test_size=0.1)
    train_filename = 'Train_{}_SNP_{}_#_{}.tsv'.format(gender,len(train.columns)-1,len(train))
    valid_filename = 'Valid_{}_SNP_{}_#_{}.tsv'.format(gender,len(valid.columns)-1,len(valid))
    train.to_csv(os.path.join(output_root,train_filename),sep='\t',index=False)
    valid.to_csv(os.path.join(output_root,valid_filename),sep='\t',index=False)
    return train, valid

# write test snp data    
# test_ped_path = '/home/obesity/input_data/ped_1112/1105male_test_ped.ped'
# output_root='/home/obesity/snp_data/snp_data_1112' 
def write_test_snp_data(ped_path, output_root, gender='Male'):
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    # read ped file
    ped = pd.read_table(ped_path, sep= '\t', low_memory=False, header= None)
    # pre-save id
    id_name = ped[0]
    # remain just snp data
    ped = ped.iloc[:,6:]
    ped.columns = ["SNP_{}".format(i) for i in range(1,len(ped.columns)+1)]
    # replace A A to AA
    replacer = Replacer(ATCG_table())
    for index in range(len(ped.columns)):
        ped.iloc[:,index] = ped.iloc[:,index].apply(replacer)
    # re-give id    
    ped.insert(loc=0, column='id', value=id_name)
    test_filename = 'Test_{}_SNP_{}_#_{}.tsv'.format(gender,len(ped.columns)-1,len(ped))
    ped.to_csv(os.path.join(output_root,test_filename),sep='\t',index=False)
    return ped