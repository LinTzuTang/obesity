import pandas as pd
from sklearn.metrics import accuracy_score,accuracy_score,f1_score,matthews_corrcoef,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
from matplotlib import gridspec

# evalute metric (accuracy,precision,sensitivity,specificity,f1,mcc)
def metric_array(test_data, test_labels, model):
    labels_score = model.predict(test_data)
    accuracy = accuracy_score(test_labels, labels_score.round())
    confusion = confusion_matrix(test_labels, labels_score.round())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP / float(TP + FP)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    f1 = f1_score(test_labels, labels_score.round())
    mcc = matthews_corrcoef(test_labels, labels_score.round()) 
    metric = [accuracy,precision,sensitivity,specificity,f1,mcc]
    return metric

# plot training progress
def show_histroy(csvlog_filepath, plot_filepath):
    df = pd.read_csv(csvlog_filepath)
    fig1 = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2) 
    ax1 = fig1.add_subplot(gs[0,0])
    ax2 = fig1.add_subplot(gs[0,1])
    #
    ax1.set_title('Train Accuracy',fontsize = '14' )
    ax2.set_title('Train Loss', fontfamily = 'serif', fontsize = '18' )
    ax1.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax1.set_ylabel('Acc', fontfamily = 'serif', fontsize = '13' )
    ax2.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax2.set_ylabel('Loss', fontfamily = 'serif', fontsize = '13' )
    ax1.plot(df['accuracy'], label = 'train',linewidth=2)
    ax1.plot(df['val_accuracy'], label = 'validation',linewidth=2)
    ax2.plot(df['loss'], label = 'train',linewidth=2)
    ax2.plot(df['val_loss'], label = 'validation',linewidth=2)
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(plot_filepath)
    plt.show()

# show train history
#show_train_history(t_m ,'accuracy','val_accuracy')
#show_train_history(t_m ,'loss','val_loss')
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
