import numpy as np
import matplotlib.pyplot as plt
import json

class Logistic_Regression:
    # create constructer
    def __init__(self,X,y,hypo_level=1, alpha=0.001, lamda=0.2, mixing=False, regularized=False,trainprob=0.7,thetas_source=None):
        if(thetas_source!=None):# continue learning
            # load info from json file
            with open(thetas_source, "r") as f:
                information = json.load(f)

                self.thetas =       information["thetas"]               ##
                self.thetas_num =   len(self.thetas)                    ##
                self.hypo_level =   information["hypo_level"]           #
                self.mixing_formats=information["mixing_formats"]       ##
                self.alpha =        information["alpha"]                #
                self.lamda =       information["lamda"]               #
                self.mixing =       bool(information["mixing"])         #
                self.regularized =  bool(information["regularized"])    #
                self.fet_num =      information["fet_num"]              #
                self.trainprob=     information["trainprob"]
                itr=     information["iteration"]
                self.thetas_start=self.thetas
                self.J_start=information["J"]
        else:# first learning(new)
            # set values from parameters in constructer
            self.fet_num = X.shape[1]
            self.hypo_level = hypo_level
            self.mixing = mixing
            self.regularized = regularized
            self.alpha = alpha
            self.lamda = lamda
            self.trainprob=trainprob

        # set y eitherway
        self.y=y

        if(not self.mixing):
            if(thetas_source==None):# first learning(new)
                self.thetas_num = self.fet_num * hypo_level + 1 # calc thetas number (1=>theta0)
            # create termes from features (new Xs)
            if(hypo_level==1):
                self.X=X
            else:                
                for i in range(1, self.hypo_level + 1):
                    self.X = np.concatenate([X,np.power(X,i)],axis=1)# x1,x2,...    x1^2,x2^2,...     x1^level,x2^level,...

        else:# use mixing
            if(thetas_source==None):# first learning(new)
                self.mixing_formats = "" #this variable take the list of indexfeatures multiplied by each other
                
                # loop in levels and get formats for every level 
                for i in range(1, hypo_level+1):
                    self.mixing_formats += self.get_possible_mixing_formats(self.fet_num, i)
                self.mixing_formats = np.array([format[:-1].split(",") for format in self.mixing_formats[:-1].split("\n")], dtype="object")# convert mixing format to list
               
                self.thetas_num = len(self.mixing_formats) + 1 # calc thetas number (1=>theta0)
                print(self.mixing_formats)
               

            # create termes from mixing format (new Xs)
            p=np.ones((X.shape[0],1))# p->calculate prduit for every format
            for i in self.mixing_formats:
                for j in i:
                    p=np.multiply(p,X[:,int(j)])
                X=np.concatenate([X,p],axis=1)
                p[:,0]=1 # inisialize 1 for new term
            self.X=X[:,self.fet_num:] # delete initial feautures (because already exist in the terms)
            # x1,x2,x3,x1x2,x1x3,x2x3,x1^2,x2^2,x3^2,x1^2x2,x1^2x3,x1^2x3....
        

        # data splitting (train/test)
        fin=(round(self.trainprob*self.X.shape[0]))

        # Train
        self.X_train=self.X[0:fin]
        self.y_train=self.y[0:fin]

        # test
        self.X_test=self.X[fin-1:,:]
        self.y_test=self.y[fin-1:,:]

        # create the thetas with random values #
        if(thetas_source==None):#first learning(new)
            self.thetas=np.random.uniform(-1,1,size=self.thetas_num)
            # self.thetas=np.zeros(self.thetas_num)
        else:# thetas already set from json file
            # call gradient and set Remaining steps to complete her work
            self.gradient_descent(itr,100)


    # this function return formats for every level
    def get_possible_mixing_formats(self, fet_num, level, mixing_string=""):
        if(level == 0):
            return mixing_string + "\n"
        result = ""
        for i in range(fet_num, 0, -1):
            result += self.get_possible_mixing_formats(
                i, level - 1, mixing_string + str(i-1)+",")
        return result

    # this function calculate Z that is sent to the sigmoid function to calculate hypothesis h(x)
    def predict(self):   
        return np.sum(self.X_train * (np.matrix(self.thetas[1:])).T )+ self.thetas[0]
    
    # and this is the sigmoid function
    def sigmoid(self, x):
        # its values ​​were limited between -700 and 36
        # Because outside this range it gives us values ​​of 0 or 1
        # So it works as a problem when calculate cost function j becauce log(0)=? and log(1-1)=?
        return 1/(1+np.exp(-(36)))  if x>36 else 1/(1+np.exp(-(-709))) if x<709 else 1/(1+np.exp(-x))
                # result = 1/(1+np.exp(-x))
                # return 0.99999 if result==1 else 0.00001 if result==0 else result

    # this is cost function J
    def J(self):
        m = len(self.y)
        regularization = (self.lamda/2*m) * np.sum( np.power(self.thetas[1:],2)  ) if self.regularized else 0 #If it is sent, it is used 
        result = -(np.sum (np.multiply(self.y , np.log( self.sigmoid(self.predict()) )) + np.multiply((1-self.y),(np.log( 1-self.sigmoid(self.predict()) )) ))/(m) )+ regularization
        return result

    # updating thetas 
    def gradient_descent(self,iteration_num=1000, saving_rate=200):
        print(self.test())
        cost=np.zeros(iteration_num)
        m,n=self.X_train.shape
        temp=np.zeros(self.thetas_num)
        for iteration in range(iteration_num):
            error=self.sigmoid(self.predict())-self.y_train
            temp[0] = ((self.alpha/m) * np.sum(error) )
            for j in range(1,self.thetas_num):  
                grd_reg = (((self.lamda/m)*self.thetas[j]) if self.regularized else 0) #If it's sent, it's used 
                # thetas:      theta_0 theta_1 theta2 theth_3 theta_4..........theta_n
                # thetas_index:   0       1      2       3      4    ..........   n
                # Xs:                     X1     X2      X3     X4   ..........   Xn
                # Xs_index:               0      1       2      3    ..........   n-1
                temp[j] =((self.alpha /m) * np.sum(np.multiply(error,self.X_train[:,j-1])))+grd_reg
            self.thetas-=temp
            cost[iteration]=self.J()
            # print("accurancy[",iteration,"]=",self.test())
            if (iteration+1) % saving_rate == 0:# informations for every 200 iteration by default
                with open("thetas_value.json", "w") as f:
                    f.write("{\n\t")
                    thetas_str=str(self.thetas).replace(" ",",").replace(",,",",").replace(",,",",")
                    f.write('"thetas":'+thetas_str+",\n\t")
                    f.write('"hypo_level":'+str(self.hypo_level)+",\n\t")
                    if (self.mixing):
                        mx=[]
                        for k in self.mixing_formats:
                            mx.append([i for i in k] )

                    

                    mxf=str(mx) if self.mixing else '""'
                    f.write('"mixing_formats":'+mxf.replace("'",'"')+",\n\t")
                    f.write('"alpha":'+str(self.alpha)+",\n\t")
                    f.write('"lamda":'+str(self.lamda)+",\n\t")
                    f.write('"mixing":"'+str(self.mixing)+'",\n\t')
                    f.write('"regularized":"'+str(self.regularized)+'",\n\t')
                    f.write('"fet_num":'+str(self.fet_num)+",\n\t")
                    f.write('"iteration":'+str(iteration_num-iteration-1)+",\n\t")
                    f.write('"trainprob":'+str(self.trainprob)+",\n\t")
                    f.write('"J":"'+str(self.J())+'"'+",\n\t")
                    f.write('"Accurancy":"'+str(self.test())+'"')

                    f.write("\n}")
                f.close()
                print("iteration Saved",iteration)
        fig,ax = plt.subplots()
        ax.plot(np.arange(iteration_num), cost, 'b')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')

    # this function calculates the percentage of prediction accuracy 
    def test(self,thetas=None):
        if (thetas==None):
            thetas=self.thetas
        v=self.X_test*((np.matrix(thetas[1:]))).T+thetas[0]
        score = 0
        error=0
        TP,TN,FP,FN=0,0,0,0
        for x, y in zip(v, self.y_test):
            if y== round(self.sigmoid(x[0,0])):
                score+=1
                if y==1:
                    TP+=1
                else:
                    TN+=1
            else:
                error+=1
                if y==0:
                    FP+=1
                else:
                    FN+=1
        # print("score:",score)
        # print("TP:",TP)
        # print("TN:",TN)
        # print("error=",error)
        # print("FP:",FP)
        # print("FN:",FN)
        # Precision = TP/(TP+FP)
        # Recall    = TP/(TP+FN)
        # F1Score = 2*(Recall * Precision) / (Recall + Precision)
        # print("F1 Score =",F1Score)
        # mtx=[[TN,FN],[FP,TP]]
        # import seaborn as sns
        # sns.heatmap(mtx,annot=True,center=True)

        return ((score)/(score+error))*100