import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

def main(Dataset):
    df=pd.read_csv(Dataset)
    
    
    Border="-"*80
    print(Border)
    print("Step 1 : load dataset")
    print(Border)
    
    print(df.head())
    
    
    print(Border)
    print("Step 2 : Exploratory Data analysis")
    print(Border)
    
    print("Null values in Each cloumn are:")
    print(df.isnull().sum())
    
    
    
    print(Border)
    print("Step 3 : Separate features and labels from dataset")
    print(Border)
    
    X=df.drop("LoanApproved", axis=1)
    print("Feature colums are :")
    print(X.columns)
    
    print("Label colum are :")
    Y=df["LoanApproved"]
    print(Y.name)
    
    # print(df.shape) --50x7
    
    print(Border)
    print("Step 4 : Split the data for traning and testing from dataset")
    print(Border)
    
    
    
    X_train,X_test,Y_train,Y_test=train_test_split(
        X,
        Y,
        random_state=42,
        test_size=0.2
    )
    print("Data is split sucessfully")
    # print(X_test.shape) --10x6 testing
    
    #------------Data Scaling--------------
        
        
    # decision Tree mostly works better on unscaled data no need to scale the data for decision tree
    print(Border)
    print("Step 5 : Train Decision Tree Classifier")
    print(Border)
        
    Model2=DecisionTreeClassifier(random_state=42,max_depth=3)
    Model2.fit(X_train,Y_train)
    Y_Pred2=Model2.predict(X_test)
    print("Accuracy are :",accuracy_score(Y_test,Y_Pred2)*100)
    
    
        
    Sobj=StandardScaler()
    X_train=Sobj.fit_transform(X_train)
    X_test=Sobj.transform(X_test)
    # print(X_Scaled[:5])
    

    
    
    
    
    print(Border)
    print("Step 6 : Train Logesting Regression")
    print(Border)
    
    Model1=LogisticRegression(max_iter=1000)
    Model1.fit(X_train,Y_train)
    Y_Pred1=Model1.predict(X_test)
    print("Accuracy are :",accuracy_score(Y_test,Y_Pred1)*100)
    
    
    
    
        
    print(Border)
    print("Step 7 : Train KNN Classifier")
    print(Border)
        
    Model3=KNeighborsClassifier(n_neighbors=11)
    Model3.fit(X_train,Y_train)
    Y_Pred3=Model3.predict(X_test)
    print("Accuracy are :",accuracy_score(Y_test,Y_Pred3)*100)
    
    
    print(Border)
    print("Step 8 : hard Voting")
    print(Border)
    
    Vote=VotingClassifier(
        estimators=[
            ("logistic",Model1),
            ("decision_tree",Model2),
            ("knn",Model3)
        ],
        voting="hard"
    )
    
    Vote.fit(X_train,Y_train)
    Y_Final1=Vote.predict(X_test)
    print("Accuracy are :",accuracy_score(Y_test,Y_Final1)*100)
    
        
    print(Border)
    print("Step 9 : Soft Voting")
    print(Border) 
    
    Vote1=VotingClassifier(
        estimators=[
            ("logistic",Model1),
            ("decision_tree",Model2),
            ("knn",Model3)
        ],
        voting="soft"
    )
    
    Vote1.fit(X_train,Y_train)
    Y_Final2=Vote1.predict(X_test)
    print("Accuracy are :",accuracy_score(Y_test,Y_Final2)*100)
    
    
if __name__=="__main__":
    main("Customer_Loan_Approval.csv")