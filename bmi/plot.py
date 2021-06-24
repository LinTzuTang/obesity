import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from bmi.customized_activation_function import CustomizedAct

# show BMI distribution
def show_bmi_distribution(train_labels, valid_labels, test_labels, gender='Male'):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    fig.suptitle('{} BMI Distribution'.format(gender), fontsize=15)
    ax1.set(xlabel='BMI', ylabel='Count')
    ax1.hist(np.round(train_labels,2),100)
    ax1.hist(np.round(valid_labels,2),100,color='red')
    ax1.text(40, 500, 'Train\nmin:{},\nmax:{},\nmedian:{}'.format(round(min(train_labels)), 
                                                                  round(max(train_labels)), 
                                                                  round(np.median(train_labels))), fontsize=12,color='blue')
    ax1.text(40, 200, 'Valid\nmin:{},\nmax:{},\nmedian:{}'.format(round(min(valid_labels)), 
                                                                  round(max(valid_labels)), 
                                                                  round(np.median(valid_labels))), fontsize=12,color='red')
    ax2.set(xlabel='BMI', ylabel='Count')
    ax2.hist(np.round(train_labels,2),100)
    ax2.hist(np.round(test_labels,2),100,color='orange')
    ax2.text(40, 500, 'Train\nmin:{},\nmax:{},\nmedian:{}'.format(round(min(train_labels)), 
                                                                  round(max(train_labels)), 
                                                                  round(np.median(train_labels))), fontsize=12,color='blue')
    ax2.text(40, 200, 'Test\nmin:{},\nmax:{},\nmedian:{}'.format(round(min(test_labels)), 
                                                                 round(max(test_labels)), 
                                                                 round(np.median(test_labels))), fontsize=12,color='orange')
    fig.show()
    

# show customized activation    
def show_customized_act(min_value=None, max_value=None, alpha=None):
    min_value = min_value or 10
    max_value = max_value or 60
    alpha = alpha or 0.1
    customized_act =  CustomizedAct(min_value,max_value,alpha)
    x = np.arange(min_value-20,max_value+20,dtype='float32')
    y = customized_act(x).numpy()
    plt.title('Customized Activation Function')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.plot(x,y)
    plt.show()
    
# show train history   
def show_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    fig.suptitle('Train History', fontsize=15)
    ax1.set(xlabel='Epoch', ylabel='loss')
    ax1.set_title('Mean Squared Error')
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax2.set_title('Mean Absolute Error')
    ax2.set(xlabel='Epoch', ylabel='mae')
    ax2.plot(history.history['mae'])
    ax2.plot(history.history['val_mae'])
    fig.show()

# show predicting result
def show_predicting_result(real_labels, predicted_labels):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    fig.suptitle('Predicting Results', fontsize=15)
    ax1.set(xlabel='Real', ylabel='Predict')
    ax1.set_title('Regression Plot')
    ax1.plot(real_labels,predicted_labels,'.')
    x = np.arange(15,40)
    ax1.plot(x,x)
    
    ax2.set_title('Residual Plot')
    ax2.set(xlabel='Residuals', ylabel='Count')
    ax2.hist(np.array(predicted_labels)-np.array(real_labels),20)
    
    fig.show()

# show predicting distribution
def show_predicting_distribution(real_labels, predicted_labels):
    plt.title('Predicting Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.hist([predicted_labels,real_labels],7, alpha=0.5,color=['red','blue'])
    plt.legend(['predict', 'real'], loc='upper left')
    plt.show()
