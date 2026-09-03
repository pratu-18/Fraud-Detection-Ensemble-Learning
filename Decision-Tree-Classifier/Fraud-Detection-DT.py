import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,
                             precision_score,recall_score,f1_score)
from sklearn.preprocessing import StandardScaler

Border="-"*80
def EDA(data,EDA_data):
    print(Border)
    print("Step 2 : EDA")
    print(Border)
    print("Null values from each column are :")
    print(dict(EDA_data))
    print("Shape of data are:")
    print(data.shape)
    
def Algorithm(X_train,X_test,Y_train,Y_test):
    print(Border)
    print("Step 4 : Train the model  Classification Algorithm ")
    print(Border)   
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    
    Model=DecisionTreeClassifier(random_state=42)
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    
    print("Accuracy of Model are:",accuracy_score(Y_test,Y_pred)*100)
    print("Confusion matrix are :")
    print(confusion_matrix(Y_test,Y_pred))
    print("Precision score are : ")
    print(precision_score(Y_test,Y_pred)*100)
    print("Recall are :")
    print(recall_score(Y_test,Y_pred)*100)
    print("F1_Score are :")
    print(f1_score(Y_test,Y_pred)*100)    
    
    
    
def Split(X,Y):
    print(Border)
    print("Step 3 : Split the data for training and testing")
    print(Border)   
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
    print("X_train =",X_train.shape)
    print("X_test =",X_test.shape)
    print("Y_train =",Y_train.shape)
    print("Y_test =",Y_test.shape)
    
    Algorithm(X_train,X_test,Y_train,Y_test)
    
    
    
    
    

def main(dataset):
    
    
    data=pd.read_csv(dataset)
    

    
    
    
    print(Border)
    print("Step 1: data load sucessfully ")
    print(Border)
    
    print(data.head())
    print("Feature columns are (X) :")
    X=data.drop("Fraud" ,axis=1)
    print(list(X.columns))
    Y=data["Fraud"]
    print("Label column are (Y) :",Y.name)
    
    EDA_data=data.isnull().sum()
    EDA(data,EDA_data)
    
    Split(X,Y)
    
    
    
    
    
    
    
    
    
    
    
    
if __name__=="__main__":
    main("Fraudulent_Transaction_Detection.csv")