import LogisticRegression as lr
import pandas as pd

####################### IMPORT DATA ####################### 

path="C:\\workspace\\github_projects\\heartdiseaselogisticregression\\data.csv"
data=pd.read_csv(path,header=None)

X = lr.np.matrix(data.iloc[:,0:data.shape[1]-1])
# features Scaling
y = lr.np.matrix(data.iloc[:,data.shape[1]-1:])
for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

# ###################### START NEW LEARNING AND TEST IT #######################

# create logistic object
e=lr.Logistic_Regression(X,y,2,0.1,lamda=0.2,mixing=True,regularized=True,trainprob=0.7)

#show thetas and cost function before train
print("___________ BEFORE _______________")
print(e.thetas)
print(e.J())

# use gradient dicent to train
e.gradient_descent(1000000,100)

#show thetas and cost value after train
print("___________ AFTER _______________")
print(e.thetas)
print(e.J())

# test model
print(str(e.test())+"%")
lr.plt.show()


####################### CONTINUE LEARNING #######################


# ---> please before executed this code go to json file and correct  errors if it exsists
# ---> if you want to add iterations edit it from json file
# ---> thanks, know you can run this code

# create lohistic object
# f=lr.Logistic_Regression(X,y,thetas_source="C:\\workspace\\github_projects\\heartdiseaselogisticregression\\thetas_value.json")

# # show thetas and cost function before train
# print("___________ START _______________")
# print(f.thetas_start)
# print(f.J_start)

# #show thetas and cost value after train
# print("___________ AFTER _______________")  
# print(f.thetas)
# print(f.J())

# # test model 
# print(str(f.test())+"%")

# lr.plt.show()


# ###################### TEST ONLY #######################

# thetas_source="C:\\workspace\\github_projects\\heartdiseaselogisticregression\\thetas_value.json""
# l=lr.Logistic_Regression(X,y,2,mixing=True,regularized=True,trainprob=0.7)
# import json
# with open(thetas_source, "r") as f:
#     information = json.load(f)
#     thetas =       information["thetas"]
# print(str(l.test(thetas))+"%")