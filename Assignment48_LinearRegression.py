import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pyplot as plt


def main():
    #-------------------------------Question 1-----------------------------------
    print("-------------------------------Question 1-----------------------------------")
    X=[1,2,3,4,5]
    Y=[3,4,2,4,5]
       
    sum=0
    for val in X:
        sum=val+sum
        
        
    
    X_mean=sum/len(X)
    print("X mean are :",X_mean)
    
    
    sum=0
   
    for val in Y:
        sum=val+sum
            
            
        
    Y_mean=sum/len(Y)
    print("Y mean are :",Y_mean)
        
        
       
   # m=sum(x-x_mean)(y-y_mean)/sum(x-x_mean)**2
    nemarator=0
    denominator=0 
    for i in range(5):
        
        nemarator=nemarator+(X[i]-X_mean)*(Y[i]-Y_mean)
        denominator=denominator+(X[i]-X_mean)**2
    M=nemarator/denominator  
    print("Slope of line(M) or coeficient are :",M)
        
        
    
    #intercept /C--> Y_mean=m*X_mean+C
    #C=m*X_mean-Y
    
    C=Y_mean-M*X_mean
    print("Intercept are (C):",C)
    
    #Y=mX+C
    #X=6 in  assignment question 
    
    Y_pred=M*6+C
    print("Predicted Y for X=6 are :",Y_pred)
    
    
    
    #-------------------------Qustion 2------------------------------
    
    print("-------------------------------Question 2-----------------------------------")
    #predic Y for all X in dataset
    
    Y_tested=[]
    for i in range(len(X)):
        Y_preds=M*X[i]+C
        print(f"Predicted Y values when X={X[i]} : {Y_preds}")
        Y_tested.append(Y_preds)
        
    print("MSC are :",mean_squared_error(Y,Y_tested)) # it contains array lists
    print("R2 score are :",r2_score(Y,Y_tested))
        
    #----------------------Question 3---------------------
    print("-------------------------------Question 3-----------------------------------")
    
    X=[1,2,3,4,5]
    Y=[20000,25000,30000,35000,40000]
    
    
    print()
    numarator=0
    dem=0
    sum=0
    
    for val in X:
        sum=sum+val
        
    X_bar=sum/len(X)
    # print(X_bar)
    
    for val in Y:
        sum=sum+val
        
    Y_bar=sum/len(Y)
    # print(Y_bar)
        
    for i in range(len(X)):
        numarator=numarator+(X[i]-X_bar)*(Y[i]-Y_bar)
        dem=dem+(X[i]-X_bar)**2
        
    M=numarator/dem
    print("M are :",M)
    
   #C=Y_bar-M*X_bar
    C=Y_bar-M*X_bar
    print("C are :",C)
    
    # now consider X=6 in question in assignmnet
    
    #Y=m*X+C
    
    Y_predSalary=M*6+C
    print("Predicted salary for 6 years experince are :",Y_predSalary)
    print("||Graph Representation of Salary||")
    
    plt.scatter(
        X,
        Y,
        linewidths=1.0,
        alpha=0.8,
        label="Salary with experince"
    )
    plt.plot(
            X,
            Y,
            # color="b",
            label="Regression Line"
        )
    
    plt.grid(True)
    plt.legend()
    plt.xlabel("Years of experince")
    plt.ylabel("Salary")
    plt.show()
    
       
        
    
    
    
    
    
    
    
if __name__=="__main__":
    main()
