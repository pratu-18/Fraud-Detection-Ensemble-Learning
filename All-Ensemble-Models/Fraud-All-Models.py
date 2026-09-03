import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,RandomForestClassifier




Border="-"*80


def Random_Forest(X_train,X_test,Y_train,Y_test):
    
    
    
    
    Model=RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        
    )
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    print("Accuracy using random forest are ",accuracy_score(Y_test,Y_pred)*100)




def Decision_Tree(X_train,X_test,Y_train,Y_test):
    
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    
    Model=DecisionTreeClassifier(random_state=42)
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    
    print("Accuracy using Decision tree are :",accuracy_score(Y_test,Y_pred)*100)



def Bagging(X_train,X_test,Y_train,Y_test):
   
    
    Base_Model=LogisticRegression(max_iter=1000)
    
    Model=BaggingClassifier(
        estimator=Base_Model
        ,n_estimators=50,
        random_state=42
        
    )
    
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    
    print("Accuracy using bagging ",accuracy_score(Y_test,Y_pred)*100)
    


def Boosting(X_train,X_test,Y_train,Y_test):
   
    
    Base_Model=LogisticRegression(max_iter=1000,random_state=42)
    
    Model=AdaBoostClassifier (
        estimator=Base_Model
        ,n_estimators=52,
        random_state=42,
        learning_rate=1.0
        
    )
    
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    
    print("Accuracy using boosting are ",accuracy_score(Y_test,Y_pred)*100)



def Hard_Voting(X_train,X_test,Y_train,Y_test):
   
    
    Base_Model1=LogisticRegression(max_iter=1000,random_state=42)
    Base_Model2=DecisionTreeClassifier(random_state=42)
    Base_Model3=KNeighborsClassifier(n_neighbors=5)
    

    Model=VotingClassifier(
    estimators=[
                ("logistic",Base_Model1),
                ("decision_tree",Base_Model2),
                ("knn",Base_Model3)
                ],
    voting="hard"
    
)
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    
    print("Accuracy using hard voting ",accuracy_score(Y_test,Y_pred)*100)
    
   
    
    
def Soft_Voting(X_train,X_test,Y_train,Y_test):
    
    
    Base_Model1=LogisticRegression(max_iter=1000,random_state=42)
    Base_Model2=DecisionTreeClassifier(random_state=42)
    Base_Model3=KNeighborsClassifier(n_neighbors=5)
    

    Model=VotingClassifier(
    estimators=[
                ("logistic",Base_Model1),
                ("decision_tree",Base_Model2),
                ("knn",Base_Model3)
                ],
    voting="soft"
    
)
    
    sobj=StandardScaler()
    X_train_scaled=sobj.fit_transform(X_train)
    X_test_scaled=sobj.transform(X_test)
    
    Model.fit(X_train_scaled,Y_train)
    Y_pred=Model.predict(X_test_scaled)
    
    print("Accuracy using soft voting ",accuracy_score(Y_test,Y_pred)*100)
    
         
    
    
    
    



def Split(X,Y):
    print(Border)
    print("Step 3 : Data spliting")
    print(Border)
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,
                                                   random_state=42,
                                                   test_size=0.2,
                                                   stratify=Y)
    print("X_train =",X_train.shape)
    print("X_test =",X_test.shape)
    print("Y_train =",Y_train.shape)
    print("Y_test =",Y_test.shape)
    
    print(Border)
    print("Step 4 : Model training")
    print(Border)
    
    Decision_Tree(X_train,X_test,Y_train,Y_test)
    Random_Forest(X_train,X_test,Y_train,Y_test)
    Bagging(X_train,X_test,Y_train,Y_test)
    Boosting(X_train,X_test,Y_train,Y_test)
    Hard_Voting(X_train,X_test,Y_train,Y_test)
    Soft_Voting(X_train,X_test,Y_train,Y_test)
    
    
    




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



def dataload(dataset):
    
    print(Border)
    print("Step 1 : data loading and details of data")
    print(Border)
    
    data=pd.read_csv(dataset)
    print(data.head())
    
    
    EDA(data)






def main():
    dataload("Fraudulent_Transaction_Detection.csv")
    
 
    
if __name__ =="__main__":
    main()