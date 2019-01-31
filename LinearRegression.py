# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:07:41 2019

@author: Sylvain J. Charbonnel
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MyModel_LR(object):
   """ 
   inputs:
   X = np.array for data parameter
   Y = np.array for data target
   trn_frac = fraction of data used for training. The rest is for validarion
   type = type of regression, default is linear, 'log' for logarithmic
   order: to be coded"""
   
   def __init__(self, X, Y, trn_frac = 0.35, order = 1, type = 'linear', norm = 'mean', reg = 'L2', regcoef = 1, lrnrate = 0.01, traintech = 'GD', GDiter = 100):
       self.__type     = type
       self.__normtype = norm
       self.__regtype  = reg
       self.__L2coef   = regcoef
       self.__tech     = traintech
       self.__lrnrate  = lrnrate
       self.__GDiter   = GDiter


       self.__N        = X.shape[0]
       self.__numparam = X.shape[-1]
       
       self.__Ntr      = int(trn_frac*self.__N)
       self.__trset    = np.random.choice(self.__N,self.__Ntr,replace=False)
       self.__Xtr_raw  = self.__makeXset__(X,self.__trset, order)
       self.__Ytr      = Y[self.__trset] #self.__normalize__(Y[self.__trset])

       self.__substract, self.__denom = self.__normalizeinfo__()
       
       self.__Xtr      = self.__normalize__(self.__Xtr_raw)

       self.__Nval     = self.__N - self.__Ntr
       self.__valset   = np.setdiff1d(np.arange(self.__N), self.__trset)
       self.__Xval_raw = self.__makeXset__(X, self.__valset, order)
       self.__Yval     = Y[self.__valset]

       self.__Xval     = self.__normalize__(self.__Xval_raw)

       self.__coef     = self.__training__()
       self.__Yhat_tr  = self.__predict__(self.__Xtr)
       self.__Yhat_val = self.__predict__(self.__Xval)

       self.__fit_tr   = self.__cost_fnc__(self.__Yhat_tr, self.__Ytr)
       self.__fit_val  = self.__cost_fnc__(self.__Yhat_val, self.__Yval)
       self.__Rsq_tr   = self.__Rsquare__(self.__fit_tr, self.__Ytr)
       self.__Rsq_val  = self.__Rsquare__(self.__fit_val, self.__Yval)
       self.__cov_tr   = self.__cov_calc__(self.__fit_tr, self.__Ntr)
       self.__cov_val  = self.__cov_calc__(self.__fit_val, self.__Nval)
       self.printinfo()

   def __normalizeinfo__(self):
      if (self.__normtype == 'none'):
         substract = 0
         denom     = 1
      elif (self.__normtype == 'range'):
         substract  = np.amin(self.__Xtr_raw,axis=0)
         denom      = np.amax(self.__Xtr_raw,axis=0) - np.amin(self.__Xtr_raw,axis=0)
         substract[0] = 0
         denom[0]     = 1
      else:
         substract    = self.__Xtr_raw.mean(axis=0)
         denom        = self.__Xtr_raw.std(axis=0)
         substract[0] = 0
         denom[0]     = 1
      return substract, denom

   def __makeXset__(self,X, fancy_idx, order):
      dim  = X[fancy_idx].shape[0]
      Xset = np.hstack((np.ones((dim,1)),X[fancy_idx]))
      for i in range(2,order+1):
         Xset = np.hstack((Xset, np.power(X[fancy_idx],i)))
      return Xset

   def __training__(self):
      y   = self.__Ytr
      dim = self.__Xtr.shape[-1]
      if (self.__type == 'log'):
          y = np.log(self.__Ytr)
          
      B = (self.__Xtr.T.dot(y)).reshape((dim,))
      A = self.__Xtr.T.dot(self.__Xtr)
      if (self.__tech == 'exact'):
         if (self.__regtype == 'L2'):
            reg_cor = self.__L2coef * np.eye(dim)
         elif (self.__regtype == 'L1'):
            reg_cor = 0 * np.eye(dim)
            print('L1 regularization is not supported with an exact solver.')
            print('Regularization has been turned off')
         else:
            reg_cor = 0 * np.eye(dim)
         self.__coef = np.linalg.solve(A,B)
      elif (self.__tech == 'GD'):
         self.__coef = np.zeros(dim)
         for i in range(self.__GDiter):
            if (self.__regtype == 'L2'):
               reg_cor = self.__L2coef * np.eye(dim).dot(self.__coef)
            elif (self.__regtype == 'L1'):
               reg_cor = self.__L2coef * np.sign(self.__coef)
            else:
               reg_cor =  np.zeros(dim)
            self.__coef = self.__coef - self.__lrnrate * (reg_cor + self.__Xtr.T.dot(self.__Xtr.dot(self.__coef)) - B)
      return self.__coef[:,np.newaxis]

   def __normalize__(self, X):
      norm_X = (X - self.__substract)/self.__denom
      return norm_X

   def __predict__(self, data):
      Yhat = data.dot(self.__coef)
      if (self.__type == 'log'):
         Yhat = np.exp(Yhat)
      return Yhat 

   def calcute_output(self, data, order):
      X = self.__makeXset__(data, np.arange(len(data)), order)
      return X.dot(self.__coef)

   def __cost_fnc__(self, Yhat, Ydata):
      fit = np.dot((Yhat - Ydata).T, Yhat - Ydata)
      return fit

   def __cov_calc__(self, fit, dim):
      cov = np.sqrt(fit)/(dim-1)
      return cov

   def __Rsquare__(self, fit, Ydata):
      Rsq = 1 - fit/((Ydata - Ydata.mean()).T.dot(Ydata - Ydata.mean()))
      return Rsq

   def get_Yhat(self, set = 'both'):
      if (set == 'training'):
         return self.__Yhat_tr
      elif (set == 'validation'):
         return self.__Yhat_val
      else:
         return self.__Yhat_val, self.__Yhat_tr

   def get_fitness(self, set = 'both'):
      if (set == 'training'):
         return self.__fit_tr[0,0]
      elif (set == 'validation'):
         return self.__fit_val[0,0]
      else:
         return self.__fit_val[0,0], self.__fit_tr[0,0]

   def get_coef(self):
      return self.__coef[:,0]

   def get_Rsq(self, set = 'both'):
      if (set == 'training'):
         return self.__Rsq_tr[0,0]
      elif (set == 'validation'):
         return self.__Rsq_val[0,0]
      else:
         return self.__Rsq_val[0,0], self.__Rsq_tr[0,0]

   def get_coef_of_variance(self, set = 'both'):
      if (set == 'training'):
         return self.__cov_tr[0,0]
      elif (set == 'validation'):
         return self.__cov_val[0,0]
      else:
         return self.__cov_val[0,0], self.__cov_tr[0,0]

   def plot_results(self, ax1 = 1, ax2 = 2, set = 'validation'):
      if (set =='training'):
         X    = self.__Xtr_raw
         Y    = self.__Ytr
         Yhat = self.__Yhat_tr
      elif (set == 'validation'):
         X    = self.__Xval_raw
         Y    = self.__Yval
         Yhat = self.__Yhat_val

      if (self.__numparam == 1):
         ax = plt.axes()
         ax.scatter(X[:,1],Y)
         ax.scatter(X[:,1],Yhat, c='r')
         
      else:
         ax = plt.axes(projection = '3d')
         ax.scatter(X[:,ax1],X[:,ax2],Y)
         ax.scatter(X[:,ax1],X[:,ax2],Yhat, c = 'r')
      plt.show()

   def printinfo(self):
       print('regression: ', self.__type)
       print('data normalization: ', self.__normtype)
       print('regularization: ', self.__regtype)
       print('solving method: ', self.__tech)

   def print_outcomes(self, set = 'validation'):
      print('fitness:', self.get_fitness(set))
      print('R^2 = ', self.get_Rsq(set))
      print('coefficient of variance:', self.get_coef_of_variance(set))
      print('Coefficients:',self.get_coef())

""" test vector
   1 = linear, exact/GD, order 1, 1 param
   2 = log, exact/GD, order 1, 1 param
   3 = linear, exact, order 1, multi param
   4 = same as 3 with Gradient descent solver
   5 = linear, exact, order 2, 1 param
   6 = same as 5 but overfitting with order 8
   7 = linear, exact/GD, order 1, 1 param, L2
   8 = same as 7 with +ve outliers
   9 = same as 7 with -ve outliers
   10= linear, exact/GD, order 1, 50 param
   11= same as 10 with L1 regularization
   12= same as 11, different weight
   13= linear,GD, order 7, L1 reg, multiparam
   14= same as 13 but normalized by std
   15 = same as 13 but notmalized by range

def __init__(self, X, Y, trn_frac = 0.35, order = 1, type = 'linear', norm = 'mean', reg = 'L2', regcoef = 1, lrnrate = 0.01, traintech = 'GD', GDiter = 100):

"""
test_vec = 15
solv = 'GD'

if (test_vec == 1):
   # linear regression:
           # Exact solver (as opposed to Gradient Descent)
           # no normalization, no regularization
           # random noise
           # no "bad" data
   numpoints = 100
   x = np.linspace(0, 10, numpoints)[:,np.newaxis]
   order = 1
   
   sigma = 4
   mu = 3
   np.random.seed(1)
   noise = np.random.normal(mu,sigma,numpoints)[:,np.newaxis]
   Y = 10.0 * x + noise


   mymodel1 = MyModel_LR(x, Y,0.15,order, 'linear', 'none','', 0, 0.0001,solv, 1000)

elif(test_vec == 2):
   # Moore law, logarithmic but linearized regression
         # Exact solver (as opposed to Gradient Descent)
         # no normalization, no regularization
         # random noise
         # no "bad" data
   numpoints = 100
   x = np.linspace(0, 10, numpoints)[:,np.newaxis]
   order = 1

   sigma = 4
   mu = 3
   np.random.seed(1)
   noise = np.random.normal(mu,sigma,numpoints)[:,np.newaxis]

   Y = np.power(2,x) + noise

   mymodel1 = MyModel_LR(x, Y,0.15,order, 'log', 'none','', 0, 0.0001,solv, 1000)       
        
elif(test_vec == 3 or test_vec == 4):
   print(test_vec)
   numpoints = 100
   x1 = np.linspace(0, 2, numpoints)[:,np.newaxis]
   x2 = np.linspace(1, 6,numpoints)[:,np.newaxis]
   x3 = np.linspace(1,6,numpoints)[:,np.newaxis]
   X = np.hstack((x1, x2,x3))
   order = 1

   sigma = 4
   mu = 3
   np.random.seed(1)
   noise = np.random.normal(mu,sigma,numpoints)[:,np.newaxis]

   Y  = 0.02*np.tanh(x1) + 0.02*np.sqrt(x1/10) + 0.05* x1 + 10 * x2 * x2 + 0.5 * x3 + noise
   
   if (test_vec == 3):
      mymodel1 = MyModel_LR(X, Y,0.15,order, 'linear', 'none','', 0, 0.0001,'exact', 1000)
   elif (test_vec == 4):
      mymodel1 = MyModel_LR(X, Y,0.15,order, 'linear', 'none','', 0, 0.0001,'GD', 1000)

elif (test_vec == 5 or test_vec == 6):
   numpoints = 100
   x = np.linspace(0, 10, numpoints)[:,np.newaxis]

   if (test_vec == 5):
      order = 2
   elif(test_vec == 6):
      order = 8

   sigma = 4
   mu = 3
   np.random.seed(1)
   noise = np.random.normal(mu,sigma,numpoints)[:,np.newaxis]
   Y = 3.2 + 10 * x - 1.7 * np.power(x,2) + noise


   mymodel1 = MyModel_LR(x, Y,0.15,order, 'linear', 'none','', 0, 0.00002,'exact', 10000)

elif (test_vec == 7 or test_vec == 8 or test_vec == 9):
   numpoints = 50
   x = np.linspace(0, 10, numpoints)[:,np.newaxis]
   order = 1


   Y = 0.5 * x + np.random.randn(numpoints)[:,np.newaxis]
   if (test_vec == 8):
      Y[-1]+=30
      Y[-2]+=30
   elif(test_vec == 9):
      Y[-1]-=30
      Y[-2]-=30

   mymodel1 = MyModel_LR(x, Y,0.8,order, 'linear', 'none','L2', 100, 0.0001,solv, 10000)
        
elif (test_vec == 10 or test_vec == 11 or test_vec == 12):
   numpoints = 100
   numparam  = 50
   x = (np.random.random((numpoints,numparam))- 0.5) * 10
   order = 1
   true_w = np.array([1, 0.5, -0.5] + [0]*(numparam-3))

   Y = x.dot(true_w) + np.random.randn(numpoints) * 0.5
   if (test_vec == 10):
      mymodel1 = MyModel_LR(x, Y,0.25,order, 'linear', 'none','L1', 0, 0.001,solv, 500)
   elif(test_vec == 11):
      mymodel1 = MyModel_LR(x, Y,0.25,order, 'linear', 'none','L1', 100, 0.001,solv, 500)
   elif(test_vec == 12):
      mymodel1 = MyModel_LR(x, Y,0.25,order, 'linear', 'none','L1', 10, 0.001,solv, 500)
   plt.plot(true_w, 'r')
   plt.plot(mymodel1.get_coef()[1:],'-b')
   plt.show()

elif (test_vec == 13 or test_vec == 14 or test_vec == 15):
   numpoints = 100
   x1 = np.linspace(0, 2, numpoints)[:,np.newaxis]
   x2 = np.linspace(1, 6,numpoints)[:,np.newaxis]
   x3 = np.linspace(1,6,numpoints)[:,np.newaxis]
   x = (np.random.random((numpoints,2))- 0.5) * 10
   X = np.hstack((x1, x2,x3,x))

   order = 6

   sigma = 1
   mu = 1
   np.random.seed(1)
   noise = np.random.normal(mu,sigma,numpoints)[:,np.newaxis]

   Y  = 0.02*np.tanh(x1) + 0.02*np.sqrt(x1/10) + 0.05* x1 + 0.2 * x2 * x2 + 0.5 * x3 + np.power(x1,4) + noise

   if (test_vec == 13):
      mymodel1 = MyModel_LR(X, Y,0.25,order, 'linear', 'none','L1', 10, 0.000001,solv, 5000)
   elif (test_vec == 14):
      mymodel1 = MyModel_LR(X, Y,0.25,order, 'linear', 'mean','L1', 10, 0.001,solv, 500)
   elif (test_vec == 15):
      mymodel1 = MyModel_LR(X, Y,0.25,order, 'linear', 'range','L1', 10, 0.001,solv, 500)       

mymodel1.plot_results(1,2,'training')
mymodel1.plot_results(1,2,'validation')

mymodel1.print_outcomes('training')
print('')
print('')
mymodel1.print_outcomes('validation')


"""



mymodel1 = MyModel_LR(x2, Y,0.45,order, 'linear', 'mean','L1', 0, 0.0001,'GD', 1000)

mymodel1.plot_results(1,2,'training')
mymodel1.plot_results(1,2,'validation')
print('fitness:', mymodel1.get_fitness('validation'))
print('R^2 = ', mymodel1.get_Rsq('validation'))
print('coefficient of variance:', mymodel1.get_coef_of_variance('validation'))
print('Coefficients:',mymodel1.get_coef())


mymodel2 = MyModel_LR(x2, Y,0.45,order, 'linear')
mymodel2.plot_results(1,2,'training')
mymodel2.plot_results(1,2,'validation')
print('R^2 = ', mymodel2.get_Rsq('validation'))
print('coefficient of variance:', mymodel2.get_coef_of_variance('validation'))
#print('Coefficients:',mymodel2.get_coef())

#print('extrapolation',mymodel2.calcute_output(np.array([[0],[8]]), order))
#print('reality')

"""
"""
tr_ratio = np.array([0.05,0.1,0.2,0.5,0.65, 0.75, 0.8, 0.9, 0.95])

cov_tr = np.array([11.3, 7.8, 5.23, 3.8, 3.3, 3.1, 3, 2.8, 2.7])
cov_val = np.array([3.15, 3.01, 3.23, 3.8, 4.4, 5.1, 5.8, 8.2, 12])

fig = plt.figure(4)
plt.plot(tr_ratio, cov_tr, '-r')
plt.plot(tr_ratio, cov_val, '-b')
plt.show()

"""
