# 10 cross validation
import pandas as pd 
import numpy as np
import os
import tensorflow.keras
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from .model import dense_model
from .model_evaluation import metric_array, show_histroy

def cross_validation(train_data,train_labels, model_output_root, fold=10, gpu=3):  
    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES']='%s'%gpu
    
    # model output path
    model_output=model_output_root
    if not os.path.exists(model_output):
        os.mkdir(model_output)
    
    # initialize
    metric_dict_final = {'metric':['accuracy','precision','sensitivity','specificity','f1','mcc']}
    metric_dict_best = {'metric':['accuracy','precision','sensitivity','specificity','f1','mcc']}
    count = 1
    
    # train models
    kfold = StratifiedKFold(n_splits=fold, shuffle=False)
    for train_index, test_index in kfold.split(train_data,train_labels):
        
        # load model
        model = dense_model(train_data)
        
        # final model setting
        final_filepath =  os.path.join(model_output,'model/model_final_%s.h5'%count)
        if not os.path.exists(os.path.dirname(final_filepath)):
            os.mkdir(os.path.dirname(final_filepath))
        saveFinalModel = tensorflow.keras.callbacks.ModelCheckpoint(final_filepath, 
                                                         verbose=1,
                                                         save_best_only=False)
        # best model setting
        best_filepath =  os.path.join(model_output,'model/model_best_%s.h5'%count)
        if not os.path.exists(os.path.dirname(best_filepath)):
            os.mkdir(os.path.dirname(best_filepath))
        saveBestModel = tensorflow.keras.callbacks.ModelCheckpoint(best_filepath, 
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True)
        # csvlogger setting
        csvlog_filepath = os.path.join(model_output,"history/PC6_csvLogger_%s.csv"%count)
        if not os.path.exists(os.path.dirname(csvlog_filepath)):
            os.mkdir(os.path.dirname(csvlog_filepath))
        CSVLogger = tensorflow.keras.callbacks.CSVLogger(csvlog_filepath,separator=',', append=False)

        # reduce learning rate
        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=50,verbose=1)
        # model fit
        t_m = model.fit(train_data[train_index], train_labels[train_index],
                        validation_data=(train_data[test_index],train_labels[test_index]),
                        epochs=300, 
                        batch_size=500,
                        callbacks=[CSVLogger,saveFinalModel,saveBestModel,reduce_lr])
        # plot 
        plot_filepath = os.path.join(model_output,"history/model_plot_%s.png"%count)
        show_histroy(csvlog_filepath, plot_filepath)
        
        # model evaluate
        # final model
        s=metric_array(train_data[test_index],train_labels[test_index],model)
        metric ={"Model_%s"%count:s}
        metric_dict_final.update(metric)
        print('final:',metric)
        
        # best model
        model = load_model(best_filepath)
        s=metric_array(train_data[test_index],train_labels[test_index],model)
        metric ={"Model_%s"%count:s}
        metric_dict_best.update(metric)
        print('best:',metric)
        
        count=count+1 # to next round
        
    # save cross validation result
    df = pd.DataFrame.from_dict(metric_dict_final, orient='columns', dtype=None).set_index('metric')
    df.to_csv(os.path.join(model_output,'final_metric.csv'))
    
    df = pd.DataFrame.from_dict(metric_dict_best, orient='columns', dtype=None).set_index('metric')
    df.to_csv(os.path.join(model_output,'best_metric.csv'))
    
    # show best models average accuracy
    acc= list(df.loc['accuracy'])*100
    with open(model_output+'/avg_acc.txt', 'w') as f:
        print('  # Accuracy: %.2f+/-%.2f' % (np.mean(acc), np.std(acc))+'\n', file = f)