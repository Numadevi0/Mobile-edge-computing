from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, RepeatVector

main = tkinter.Tk()
main.title("Deep Learning for Secure Mobile Edge Computing in Cyber-Physical Transportation Systems") #designing main screen
main.geometry("1300x1200")

global filename
global precision, recall, fscore, accuracy
global X, Y
global dataset
global X_train, X_test, y_train, y_test
global dbn_model, labels, scaler

def uploadDataset(): #function to upload dataset
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="AndroidDataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    dataset = pd.read_csv(filename, sep=";")
    text.insert(END,str(dataset))
    labels, count = np.unique(dataset['type'].ravel(), return_counts = True)
    labels = ['Non-Attack', 'Attack']
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks()
    plt.tight_layout()
    plt.show()

def preprocessDataset():
    global dataset, X, Y, scaler
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    #normalizing dataset values
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)#normalize train features
    text.insert(END,"Normalized Android APK extracted features\n\n")
    text.insert(END,str(X))

def testSplit():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #dataset preprocessing to shuffle values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffling dataset values
    X = X[indices]
    Y = Y[indices]
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Total records found in dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset= "+str(X.shape[1])+"\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.tight_layout()
    plt.show()    

def runSVM():
    text.delete('1.0', END)
    global precision, recall, fscore, accuracy
    global X_train, X_test, y_train, y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    #training and evaluating performance of SVM algorithm
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = svm_cls.predict(X_test) #perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("SVM Algorithm", y_test, predict)

def runDecisionTree():
    global precision, recall, fscore, accuracy
    global X_train, X_test, y_train, y_test
    #training and evaluating performance of Decision Tree algorithm
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = dt_cls.predict(X_test) #perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Decision Tree Algorithm", y_test, predict)
    
def runRandomForest():
    global precision, recall, fscore, accuracy
    global X_train, X_test, y_train, y_test
    #training and evaluating performance of Random Forest algorithm
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = rf_cls.predict(X_test) #perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Random Forest Algorithm", y_test, predict)

def runProposeDBN():
    global precision, recall, fscore, accuracy
    global X_train, X_test, y_train, y_test, dbn_model
    #training CNN deep learning algorithm to predict factory maintenaance
    #converting dataset shape for CNN comptaible format as 4 dimension array
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #creating backpropagation Deep Belief neural network object
    dbn_model = Sequential()
    #defining dbN layer to select and filtered features from the dataset
    dbn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    #defining maxpool layet to collect relevant filtered features from previous features layer layer
    dbn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #creating another DBN layer with 16 neurons to optimzed features 16 times
    dbn_model.add(Convolution2D(16, (1, 1), activation = 'relu'))
    #max layet to collect relevant features
    dbn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #backpropagation layer to convert multidimension features to single flatten size
    dbn_model.add(Flatten())
    #define output prediction layer with labels to predict
    dbn_model.add(Dense(units = 256, activation = 'relu'))
    dbn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    #compile, train and load CNN model
    dbn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/dbn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/dbn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = dbn_model.fit(X_train1, y_train1, batch_size = 8, epochs = 50, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/dbn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        dbn_model.load_weights("model/dbn_weights.hdf5")
    #perform prediction on test data   
    predict = dbn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Propose Backpropogation DBN Algorithm", y_test1, predict)

def runCNN():
    global precision, recall, fscore, accuracy
    global X_train, X_test, y_train, y_test
    
    #training CNN deep learning algorithm to predict factory maintenaance
    #converting dataset shape for CNN comptaible format as 4 dimension array
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)

    cnn_model = Sequential()
    cnn_model.add(InputLayer(input_shape=(X_train1.shape[1], X_train1.shape[2], X_train1.shape[3])))
    #creating conv2d layer of 64 neurons or filters of 5 X 5 matrix to filter features
    cnn_model.add(Conv2D(64, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    #max layer to collect relevant filtered features from previous layer
    cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #defining another layer
    cnn_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    cnn_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #features normalization
    cnn_model.add(BatchNormalization())
    #adding another CNN layer
    cnn_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    cnn_model.add(MaxPool2D(pool_size=(1, 1), padding='valid'))
    cnn_model.add(BatchNormalization())
    #dropout to remove irrelevant features
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=100, activation='relu'))
    cnn_model.add(Dense(units=100, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(units=y_train1.shape[1], activation='softmax'))
    #compiling and training and loading model
    cnn_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train1, y_train1, batch_size = 8, epochs = 50, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on test data   
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("CNN Algorithm", predict, y_test1)
    

def graph():
    global precision, recall, fscore, accuracy
    #comparison graph between all algorithms
    df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','FSCORE',fscore[0]],
                       ['Decision Tree','Accuracy',accuracy[1]],['Decision Tree','Precision',precision[1]],['Decision Tree','Recall',recall[1]],['Decision Tree','FSCORE',fscore[1]],
                       ['Random Forest','Accuracy',accuracy[2]],['Random Forest','Precision',precision[2]],['Random Forest','Recall',recall[2]],['Random Forest','FSCORE',fscore[2]],
                       ['Propose DBN Backpropagation','Accuracy',accuracy[3]],['Propose DBN Backpropagation','Precision',precision[3]],['Propose DBN Backpropagation','Recall',recall[3]],['Propose DBN Backpropagation','FSCORE',fscore[3]],
                       ['CNN','Accuracy',accuracy[4]],['CNN','Precision',precision[4]],['CNN','Recall',recall[4]],['CNN','FSCORE',fscore[4]], 
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.tight_layout()
    plt.show()

def predict():
    global dbn_model, scaler, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="AndroidDataset")
    pathlabel.config(text=filename)
    dataset = pd.read_csv(filename, sep=";")
    dataset.fillna(0, inplace = True)
    data = dataset.values
    X = scaler.transform(data)
    XX = X
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    predict = dbn_model.predict(X)
    for i in range(len(predict)):
        pred = predict[i]
        pred = np.argmax(pred)
        print(pred)
        text.insert(END,"Test Data = "+str(XX[i])+" Predicted As ====> "+labels[pred]+"\n\n")
    

font = ('times', 16, 'bold')
title = Label(main, text='Deep Learning for Secure Mobile Edge Computing in Cyber-Physical Transportation Systems')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Android APK Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=360,y=100)

preprocessButton = Button(main, text="Extract & Preprocess Features", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1) 

splitButton = Button(main, text="Train & Test Split", command=testSplit)
splitButton.place(x=330,y=150)
splitButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=650,y=150)
svmButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=50,y=200)
dtButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=330,y=200)
rfButton.config(font=font1)

proposeButton = Button(main, text=" Backpropagation DBN Network", command=runProposeDBN)
proposeButton.place(x=650,y=200)
proposeButton.config(font=font1)

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=50,y=250)
cnnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=500,y=250)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict MEC Attack from Test Data", command=predict)
predictButton.place(x=650,y=250)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=19,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
