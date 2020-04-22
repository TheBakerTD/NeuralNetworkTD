import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math, time
from IPython.display import clear_output

def training_data(size):
    x=np.random.rand(size)
    y=np.random.rand(size)
    label=np.zeros(size)
    for i in range(size):
        if y[i]>x[i]:
            label[i]=+1
        else:
            label[i]=-1
    return [x,y,label]

def plot_training_data(training_data):
    for i in range(len(training_data[0])):
        if training_data[2][i]<0:
            plt.plot(training_data[0][i],
                        training_data[1][i],
                        color='k',
                        marker='o',
                        fillstyle='none',
                        )
        else:
            plt.scatter(training_data[0][i],
                        training_data[1][i],
                        color='k',
                        marker='.')

def perceptron(x,y,weight_x,weight_y):
    if (x*weight_x+y*weight_y)>0:
        return 1
    else:
        return -1
    
def correction(guess,true_value,learning_rate):
    return (true_value-guess)

def iterate(input_data, weight_x, weight_y, iteration_count,learning_rate):
    correction_history=[]
    error_history=[]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    
    for i in range(iteration_count):
        error_history=[]
        for j in range(len(input_data[0])):
            weight_x=weight_x
            weight_y=weight_y
            guess=perceptron(input_data[0][j],
                             input_data[1][j],
                             weight_x,
                             weight_y)
            
            corr=correction(guess,
                            input_data[2][j],
                            learning_rate)
            
            error_history.append(corr)
            correction_history.append([weight_x,weight_y])            
            weight_x=weight_x+corr*learning_rate*input_data[0][j]
            weight_y=weight_y+corr*learning_rate*input_data[1][j]
        
        #clear_output(wait=True)
        for j in range(len(error_history)):
            if error_history[j]==0:
                ax1.scatter(np.array(input_data[0][j]),np.array(input_data[1][j]),color='g')
            else:
                ax1.scatter(np.array(input_data[0][j]),np.array(input_data[1][j]),color='r')
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)
        
        ax1.plot([0,1],[0,(-weight_y/weight_x)])
        ax2.plot(correction_history)
        ax2.set_xlim(0,iteration_count*len(input_data[0]))
        ax1.figure
        plt.pause(0.1)
        ax1.clear()
        ax2.clear()
    return correction_history,error_history

test=training_data(25)
#plt.plot(test)
out1,out2=iterate(test,5.,-1.,50,0.01)
#plt.plot(out1)




    

