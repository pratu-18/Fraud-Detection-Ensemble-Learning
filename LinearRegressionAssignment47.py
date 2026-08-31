import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    df=pd.DataFrame({
        "StudyHours":[1,2,3,4,5],
        "Marks":[50,55,60,65,70]
    })
    
    
    #-------------------------------------Assignment 47 Question=7----------------------------
    X_train,X_test,Y_train,Y_test=train_test_split(
        df[["StudyHours"]],
        df["Marks"],
        test_size=0.2,
        random_state=42
    )
    
    Model=LinearRegression()
    Model.fit(X_train,Y_train)
    Y_pred=Model.predict(X_test)
    # print("Actual",Y_test)
    # print("Predicted",Y_pred)
    
    # print(X_train.shape)
    # print(X_test.shape)
    # print(Y_train.shape)
    # print(Y_test.shape)
    print("Coeficient are :",Model.coef_)
    print("Intercept are (C):",Model.intercept_)
    
   
    
   
    #----------------------------------------Assignment 47 Question 8------------------------------
     #Y=mX+C
     #Y=5*6+45
     #Y=75
     
    Testing=pd.DataFrame({
        "StudyHours":[6]
    })
    Res=Model.predict(Testing)
    print("If study hours are 6 then Marks are :",Res)
    
    
        #----------------------------------------Assignment 47 Question 9------------------------------
        
    data=pd.DataFrame({
        "StudyHours":[1,2,3,4,5],
        "SleepHours":[7,6,7,6,8],
        "Marks":[50,55,60,65,70]
    })

    # 
    X=data.drop("Marks", axis=1)
    Y=data["Marks"]
    # print(X)
    # print(Y)
    
    X_train,X_test,Y_train,Y_test=train_test_split(
        X,
        Y,
        random_state=42,
        test_size=0.2
    )
    # print(X_train.shape)
    # print(X_test.shape)
    
    Model1=LinearRegression()
  
    Model1.fit(X_train,Y_train)
    Y_pred=Model1.predict(X_test)
    print("Coeficient of study hours and sleep hours are: ",Model1.coef_)
    print("Intercept (C) are:",Model1.intercept_)
    

    
    
    
if __name__=="__main__":
    main()