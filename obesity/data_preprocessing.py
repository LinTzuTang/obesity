import pandas as pd
import os
import argparse
from sklearn.preprocessing import MinMaxScaler

# get patients' id and spilt to obesity & normal
def get_ids(patient_data,ids):
    part = patient_data[patient_data['1'].isin(ids)]
    obesity_ids = part[part['O']==1]['1']
    normal_ids = part[part['O']!=1]['1']
    return obesity_ids,normal_ids

# define replace table
def table():
    table = {}
    for first in list('ATCG'):
        for second in list('ATCG'):
            table[first+" "+second] = first+second
    table['0 0'] = '00'
    return table

# write SNP data based on patients' id
def write_snp_data(ped,ids,root,prefix):
    ids = ids or list(ped[0])
    if not os.path.exists(root):
         os.mkdir(root)
    seqs = ped[ped[1].isin(ids)].iloc[:,6:].replace(table())
    seqs.columns = ["SNP_{}".format(i) for i in range(1,1+len(seqs.columns))]
    seqs.index = ["{}_{}".format(prefix, i) for i in range(1,1+len(seqs.index))]
    path = '{}_SNP_{}_#_{}.tsv'.format(prefix,len(seqs.columns),len(seqs))
    seqs.to_csv(os.path.join(root,path),sep='\t',index=False)
    file_path = os.path.join(root,path)
    return file_path

# write phenotype data in ped (update on 20210503)
def write_phenotype_data(ped, output_phenotype_root, prefix, phenotype_path=None):
    if not os.path.exists(output_phenotype_root):
         os.mkdir(output_phenotype_root)
    phenotype_path = phenotype_path or '/home/obesity/input_data/TWB2_phenotype.txt'
    phenotype = pd.read_csv(phenotype_path)
    p = phenotype[phenotype['TWB2_ID'].isin(ped[0])].set_index('TWB2_ID').loc[ped[0]].reset_index()
    p.to_csv(os.path.join(output_phenotype_root, prefix+'.csv'),index = False)
    return os.path.join(output_phenotype_root, prefix+'.csv')

# input: ped file, patient data 
# output: SNP data
def write_obesity_patient_snp_data(ped_path,patient_path,output_root):
    # read ped file
    table = {}
    for first in list('ATCG'):
        for second in list('ATCG'):
            table[first+" "+second] = first+second
    table['0 0'] = '00'
    ped = pd.read_table(ped_path, sep= '\t', low_memory=False, header= None)
    ped.iloc[:,6:] = ped.iloc[:,6:].replace(table)
    # read patient data
    patient_data = pd.read_table( patient_path,sep = ',')
    
    obesity_ids,normal_ids = get_ids(patient_data,ped[1])
    obesity_data_path = write_snp_data(ped,obesity_ids, output_root, os.path.splitext(os.path.basename(ped_path))[0]+"_obesity")
    normal_data_path = write_snp_data(ped,normal_ids, output_root, os.path.splitext(os.path.basename(ped_path))[0]+"_normal")
    return obesity_data_path, normal_data_path



if __name__ == '__main__':   

    parser = argparse.ArgumentParser(description='write_patient_snp_data')
    parser.add_argument('-i','--input_ped',help='your ped file path',required=True)
    parser.add_argument('-p','--patient_data',help='MOlist of patient data',required=True)
    parser.add_argument('-o','--output_root',help='output directory',required=True)
    args = parser.parse_args()
    write_obesity_patient_snp_data(ped_path = args.input_ped, patient_path = args.patient_data, output_root = args.output_root)
    
    
# example
# python3 data_preproocessing.py --input_ped '/home/obesityinput_data/ped/TWB2_female_3060_bmi2430_exclude_combine_plus_KM_plus_giant0921_acgt_ped.ped' --patient_data '/home/obesity/input_data/MOlist.csv' --output_root '/home/obesity/snp_data'    