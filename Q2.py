import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable

result = PrettyTable()

file= open("Q2_data/X_test.pkl", "rb")
xtest=pickle.load(file)
file.close()

file= open("Q2_data/X_train.pkl", "rb")
xtrain=pickle.load(file)
file.close()

file= open("Q2_data/Y_train.pkl", "rb")
ytrain=pickle.load(file)
file.close()

file= open("Q2_data/Fx_test.pkl", "rb")
ytest=pickle.load(file)
file.close()

# print(xtest.shape)
# print(xtrain.shape)
# print(ytest.shape)
# print(ytrain.shape)

# np.random.shuffle(data)
# x=np.array(data[:,0])
# y=np.array(data[:,1])

# lim=int(0.9*x.shape[0])
# xtrain=np.array(x[0:lim])
# xtest=np.array(x[lim:x.shape[0]])
# ytrain=np.array(y[0:lim])
# ytest=np.array(y[lim:y.shape[0]])

#print(np.shape(xtrain))
# for i in range(len(xtrain)):
#     print(xtrain[i], ytrain[i])
# print(xtrain)
# print(ytrain)

# x=np.array_split(xtrain,20)
# y=np.array_split(ytrain,20)

bias_values = []
bias_values_sq = []
variance_mean = []

for degree in range(1,10):

    ypred_values = [[] for i in range(20)]
    polynomial=PolynomialFeatures(degree=degree)
    
    for model in range(0,20):
        
        xp=polynomial.fit_transform(np.array(xtrain[model]).reshape(-1,1))
                
        xl=LinearRegression()
        xl.fit(xp, ytrain[model])
        
        ypred_values[model]=(xl.predict(polynomial.fit_transform(np.array(xtest).reshape(-1,1))))
        
        # plt.scatter(x[model], y[model], color="red", label = "Actual")
        # plt.scatter(x[model], ypred, color="blue", label = "Fit")
        # plt.show()

    # plt.scatter(xtest, ytest, color="red", label = "Actual")
    # plt.scatter(xtest, ypred_values[model], color="blue", label = "Fit")
    # plt.title("Degree = " + str(degree))
    # plt.legend()
    # plt.show()

    ypred_mean = np.mean(ypred_values, axis=0)
    bias= (ypred_mean-ytest) ** 2
    
    bias_mean=np.mean(bias) ** 0.5
    bias_values.append(bias_mean)

    # Only for graph purpose
    bias_mean_sq=np.mean(bias)
    bias_values_sq.append(bias_mean_sq)
    
    var = np.var(ypred_values, axis=0)
    variance_mean.append(np.mean(var))

result.field_names=["Degree", "Bias", "Variance"]
for i in range(1,10):
    result.add_row([i, bias_values[i-1], variance_mean[i-1]])

print(result)

plt.subplot(2, 2, 1)
plt.plot(range(1,10), bias_values , color="red", label = "Bias")
plt.xlabel("Degree", fontsize = 'medium')
plt.ylabel("Bias", fontsize = 'medium')
plt.title("Bias vs. Degree")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1,10), variance_mean, color="blue", label = "Variance")
plt.xlabel("Degree", fontsize = 'medium')
plt.ylabel("Variance", fontsize = 'medium')
plt.title("Variance vs. Degree")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1,10), bias_values_sq , color="red", label = "(Bias)^2")
plt.plot(range(1,10), variance_mean, color="blue", label = "Variance") 
plt.xlabel("Complexity", fontsize = 'medium')
plt.ylabel("Error", fontsize = 'medium')
plt.title("(Bias)^2 vs. Variance")
plt.legend()
plt.show()
