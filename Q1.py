import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

file= open("Q1_data/data.pkl", "rb")
data=pickle.load(file)
file.close()

np.random.shuffle(data)
x=np.array(data[:,0])
y=np.array(data[:,1])


lim=int(0.9*x.shape[0])
xtrain=np.array(x[0:lim])
xtest=np.array(x[lim:x.shape[0]])
ytrain=np.array(y[0:lim])
ytest=np.array(y[lim:y.shape[0]])



#print(np.shape(xtrain))
# for i in range(len(xtrain)):
#     print(xtrain[i], ytrain[i])

# print(xtrain)
# print(ytrain)




#xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.1)
x=np.array_split(xtrain,10)
y=np.array_split(ytrain,10)

bias=[]
variance=[]

for degree in range(30,40):

    polynomial=PolynomialFeatures(degree=degree)
    
    for model in range(0,10):
        
        xp=polynomial.fit_transform(np.array(x[model]).reshape(-1,1))
        
        xl=LinearRegression()
        xl.fit(xp, y[model])

        plt.scatter(x[model], y[model], color="red")

        ypred=xl.predict(polynomial.fit_transform(np.array(x[model]).reshape(-1,1)))

        plt.scatter(x[model], ypred, color="blue")

        
        plt.show()






