import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import argparse
from obesity.data_preproocessing import write_obesity_patient_snp_data
from obesity.snp_encoding_traintest_split import get_balanced_encoded_train_data_and_labels
from obesity.ten_cross_validation import cross_validation

# ped_root = '/home/obesity/input_data/ped'
# patient_data = '/home/obesity/input_data/MOlist.csv'
# snp_data_output_root = '/home/obesity/snp_data'
# model_output_root = '/home/obesity/cv_results_10_fold'
# gpu = 3

def multi_files_10cv(input_ped_root, patient_data, snp_data_output_root, model_output_root, gpu):
    for ped in [file for file in os.listdir(input_ped_root) if file.endswith('.ped')]:
        obesity_data_path, normal_data_path = write_obesity_patient_snp_data(os.path.join(input_ped_root,ped), patient_data, snp_data_output_root)
        train_data, train_labels = get_balanced_encoded_train_data_and_labels(obesity_data_path, normal_data_path)
        cross_validation(train_data,train_labels, os.path.join(model_output_root,os.path.splitext(ped)[0]), fold=10, gpu=gpu)
        
        
        
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='multi_files_10cv')
    parser.add_argument('-i','--input_ped_root',help='ped files root',required=True)
    parser.add_argument('-p','--patient_data',help='MOlist of patient data',required=True)
    parser.add_argument('-s','--snp_data_output_root',help='snp data output directory',required=True)
    parser.add_argument('-m','--model_output_root',help='10cv results ouput directory',required=True)
    parser.add_argument('-g','--gpu',help='gpu(0,1,2,3)',type=int)
    args = parser.parse_args()
    multi_files_10cv(input_ped_root=args.input_ped_root, patient_data=args.patient_data, snp_data_output_root=args.snp_data_output_root, model_output_root=args.model_output_root, gpu=args.gpu)
    
    
# python3 multi_files_10cv.py -i '/home/obesity/input_data/ped' -p '/home/obesity/input_data/MOlist.csv' -s '/home/obesity/snp_data' -m '/home/obesity/cv_results_10_fold' -g 3