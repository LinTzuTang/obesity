import pandas as pd
import numpy as np
from io import StringIO

# generate SNP encoding table(dict) (AC = CA)
def integer_encode_table():
    snp_uni = ['00', 'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC','GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    integer = list(range(len(snp_uni)))
    snp_uni
    table = {}
    for index,key in enumerate(snp_uni):
        table[key] = integer[index]
    return table

# generate SNP encoding table(dict) (AC != CA)
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

# read BMI data
# bmi_data_path ='/home/obesity/input_data/ped_1112/1105_male_train.txt'
# show min & max BMI
def print_min_max_bmi(bmi_data_path):
    bmi = pd.read_csv(bmi_data_path,sep = '\t')
    print(min(bmi['BMI']), max(bmi['BMI']))
    
# to dict    
def read_bmi_data(bmi_data_path):
    return read_patient_data(bmi_data_path,'BMI')

# to dict 
def read_patient_data(bmi_data_path,value_name= 'pheno'):
    bmi = pd.read_csv(bmi_data_path,sep = '\t')
    bmi_dict = dict(zip(bmi['IID'],bmi[value_name]))
    return bmi_dict

# SNP_encoding then convert data format to np.array
def encode_snp_data_get_bmi_label(snp_data, bmi_data_path,value_name= 'BMI'):
    # copy
    snp_data_ = snp_data.copy()
    # encoding
    snp_data_.iloc[:,1:] = snp_data_.iloc[:,1:].replace(SNP_encode_table())
    # encoded sequences to nparray
    encoded_seqs = list(np.array(snp_data_.iloc[:,1:]))
    # to dict (key=id, value=encoded seq)
    encoded_seq_dict = dict(zip(snp_data_.iloc[:,0],encoded_seqs))
    # read bmi data
    bmi_dict = read_patient_data(bmi_data_path,value_name)
    # labeling
    encoded_seq_array = []
    labels = []
    for id_,encoded_seq in encoded_seq_dict.items():
        encoded_seq_array.append(encoded_seq)
        labels.append(bmi_dict[id_])
    encoded_seq_array = [list(i) for i in np.array(encoded_seq_array)]
    return np.array(encoded_seq_array), np.array(labels)


# fast encoding (updated on 20210611)
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
# SNP_fast_encoding_get_bmi
def snp_fast_encoding_and_get_bmi_label(snp_data_path,phenotype_data_path):
    # data encoding
    encoded_data = fast_encoding(snp_data_path)
    # read phenotype
    phenotype = pd.read_csv(phenotype_data_path, index_col=0)
    # give labels 
    labels = np.array(phenotype['BMI'])
    return encoded_data, labels
