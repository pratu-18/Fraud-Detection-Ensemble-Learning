import pandas 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score,recall_score,
                             f1_score)


Border="-"*80


def ModelTraning(X_train,X_test,Y_train,Y_test):
    
    print(Border)
    print("Step 4 : Model training")
    print(Border)
    
    
    
    Model=RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        
    )
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    print("Accuracy are ",accuracy_score(Y_test,Y_pred)*100)
    
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
    print("Step 3 : Data spliting")
    print(Border)
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
    print("X_train =",X_train.shape)
    print("X_test =",X_test.shape)
    print("Y_train =",Y_train.shape)
    print("Y_test =",Y_test.shape)
    
    ModelTraning(X_train,X_test,Y_train,Y_test)

def EDA(data):
    print(Border)
    print("Step 2 : EDA")
    print(Border)
    
    print("Missing values from each columns are :")
    print(list(data.isnull().sum()))
    print("Separate features(X) and labels(Y) columns")
    X=data.drop("Fraud",axis=1)
    for val in X.columns:
        print(val)
    
    Y=data["Fraud"]
    print(Y.name," -label")
    
    Split(X,Y)
    
    
    


def DataLoad(dataset):
    data=pandas.read_csv(dataset)
    print(Border)
    print("Step 1 : data loading and details")
    print(Border)
    print("Some records from data are :")
    print(data.head())
    print("Shape of data :")
    print(data.shape)
    
    EDA(data)
    



def main():
    
    DataLoad("Fraudulent_Transaction_Detection.csv")
    
    
    
if __name__=="__main__":
    main()