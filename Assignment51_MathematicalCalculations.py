import numpy as np
from math import sqrt
def main():
    data=[4,6,8,10,12]
    
    Border="-"*80
    print(Border)
    print("Given dataset are :")
    print(Border)
    print(Border)
    sum=0
    for num in data:
        sum=sum+num
    
    X_bar=sum/len(data)
    print("Mean of the dataset",X_bar)
    print(Border)
    
    print("Deviation of each items from mean(x-x_bar)")
    L1=list()
    for num in data:
        fin=num-X_bar
        print(fin)
        L1.append(fin)
        
    print(Border)
    print("Square of each calculated deviation")
    print(Border)
    L2=list()
    for val in L1:
        val=val**2
        print(val)
        L2.append(val)
        
    
    
    
    #Varience=sum(x-X_bar)**2/n
    
    sum=0
    for no in L2:
        sum=sum+no/len(data)
    print("Variance of given dataset are :",sum)
   
    
    # sum=0
    # for no in data:
    #     sum=sum+(no-X_bar)**2/len(data)
    # print(sum)
    
        
    
    data2=[5,7,9,11,13]
    print(Border)
    print("Given dataset are:")
    print(data2)
    print(Border)
    
    sum=0
    for val in data2:
        sum=sum+val
        
    mean=sum/len(data2)
    print("Mean are :",mean)
    
    sum=0
    for i in data2:
        sum=sum+(i-mean)**2/len(data2)
        
    print("Variance are :",sum)
    
    print("Standard deviation are:",sqrt(sum))
    
    print(Border)
    
    
    #calculate Standard scaler of some items from data3
    data3=[6,7,8,9,10,11,12]
    mean_data3=9
    SD=2
    
    #Standar scalar= X-X_bar/Standar Deviation
    
    print("Standard Scalar of some values from given dataset")
    for no in range(0,len(data3),3):
        Standard_scalar=(data3[no]-mean_data3)/SD
        print(Standard_scalar)
    
        
        
        
        
        
    
    
    
    
    
 
        
    
    
    
if __name__=="__main__":
    main()