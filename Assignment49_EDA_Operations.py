import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.metrics import confusion_matrix,classification_report

def main():
    #--------------------Question 1-------------------------
    arr=np.array([6,7,8,9,10,11,12])
    X_bar=arr.mean()
    print("Mean of given dataset are :",X_bar)
    
    #--------------------------Question 2------------------------
    
    numarator=0
    denominator=len(arr)

    for i in arr:
        numarator=numarator+((i-X_bar)**2)
        
        
        
    variance=numarator/denominator
    print("Variance are :",variance)
    print("Stadard deviation are :",sqrt(variance))
    
    
    #inbuild methods
    # print(np.var(arr))
    # print(np.std(arr))
      
    #--------------------------Question 3------------------------
    
    arr2=np.array(
        [[25,2000],
         [30,4000],
         [35,80000]]
    )
    stc=StandardScaler()
    
    print("Standard Scaler of given dataset are:")
    Scaled_data=stc.fit_transform(arr2)
    print(Scaled_data)
          
        
    #--------------------------Question 4------------------------
    
    Data=[
        {"X": 1 ,"Y":6},
        {"X": 2 ,"Y":5},
        {"X": 3 ,"Y":4}
    ]
    
    new_point={"X":5, "Y":9}
    print("Euclidean distance before feature scaling")
    L1=list()
    for dict in Data:
        Distance=sqrt((new_point["X"]-dict["X"])**2 + (new_point["Y"]-dict["Y"])**2)
        L1.append(Distance)
        
    for i in range(len(L1)):
        print(f"Point {i} disatnace are :",L1[i])
        
    
    print("Euclidean distance after feature scaling")

    #Covert Data to dataframe to scale the data
    df=DataFrame(Data)
    Scaled=stc.fit_transform(df)
    
    new_df=DataFrame([new_point])
    new_scaled_points=stc.transform(new_df)
    
    L2=list()
    for i in range(len(Scaled)):

        Distance = sqrt(
        (new_scaled_points[0][0] - Scaled[i][0])**2 +
        (new_scaled_points[0][1] - Scaled[i][1])**2
    )

        L2.append(Distance)
    for i in range(len(L2)):
        print(f"Point {i} distance are :", L2[i])
    
    
    #-------------------Question 7--------------------------------
    
    Actual=[1,1,1,1,0,0,0,0]
    Predicted=[1,1,0,1,0,1,0,0]
    
    print(confusion_matrix(Actual,Predicted))
    print(classification_report(Actual,Predicted))
    
    

        
    
    
        
        
    
    
        
    
    
    
if __name__=="__main__":
    main()