import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

def main(FileName):
    
    data=pd.read_csv(FileName)
    # print(data.head())
    # print(data.isnull().sum())
    X=data.drop("target",axis=1)
    # print(X.columns)
    Y=data["target"]
    # print(Y.name)
    print(data.shape)
    X_train,X_test,Y_train,Y_test=train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )
    
    # print(X_train.shape)
    
    Base_model=DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    )
    
    Base_model.fit(X_train,Y_train)
    Y_pred=Base_model.predict(X_test)
    
    # print("Actual")
    # print(Y_test[:5])
    # print("Expected")
    # print(Y_pred[:5])
    
    
    
    Border="-"*80
    print(Border)
    print(" Result using single decision tree")
    print(Border)
    
    print("MSE are(minums errors including all) :",mean_squared_error(Y_test,Y_pred)*100) #yevdhe error ale
    print("R2score are :",r2_score(Y_test,Y_pred)*100)#yevdhi accuracy
        
        
    print(Border)
    print("Result using multiple desicion tree (Random Forest regressor)")
    print(Border)
    
    Fin_Model=RandomForestRegressor(
        
        n_estimators=10,
        random_state=42
    )
    
    Fin_Model.fit(X_train,Y_train)
    Y_pred1=Fin_Model.predict(X_test)
    
    print("MSE are(minums errors including all) :",mean_squared_error(Y_test,Y_pred1)*100) #yevdhe error ale
    print("R2score are :",r2_score(Y_test,Y_pred1)*100)#yevdhi accuracy
        
    
    

    
    
    
    
    
    
if __name__=="__main__":
    main("california_housing.csv")