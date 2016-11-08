"""
Gaussisan HMM of data generated randomly
Uses Gaussian HMM on data generated randomly
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from itertools import permutations
from scipy.linalg import norm
from hmmlearn import hmm
import math
from datetime import datetime
from numpy import genfromtxt


def createSample(n,l,k):
    #n is number of samples
    #l is number of sequences
    #k is number of components

    startprob_org = np.random.rand(k)
    startprob_org = startprob_org/np.sum(startprob_org,0)
    #print "start prob ",startprob_org
    #np.array([0.6, 0.3, 0.1, 0.0])

    # The transition matrix, note that there are no transitions possible
    # between component 1 and 3
    transmat_org = np.random.rand(k,k)
    transmat_org = transmat_org / np.sum(transmat_org,1).reshape((k,1))
    #print "transmat org",transmat_org
    #transmat_org = np.array([[0.7, 0.2, 0.0, 0.1],
                 #        [0.3, 0.5, 0.2, 0.0],
                  #       [0.0, 0.3, 0.5, 0.2],
                   #      [0.2, 0.0, 0.2, 0.6]])

    # The means of each component
    means_org = np.random.rand(k,l)
        #np.array([[0.0,  0.0],
         #             [0.0, 11.0],
          #            [9.0, 10.0],
           #           [11.0, -1.0]])
    # The covariance of each component
    covars_org = .5*np.tile(np.identity(l),(k,1,1))
        #.5 * np.tile(np.identity(2), (4, 1, 1))

    # Build an HMM instance and set parameters
    model = hmm.GaussianHMM(n_components=k, covariance_type="full")
    model.startprob_ = startprob_org
    model.transmat_ = transmat_org
    model.means_ = means_org
    model.covars_ = covars_org

    # Generate samples
    X, Z = model.sample(n)

    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6 * X.shape[0]))

    # separate out training and testing data
    trainX = X[:train_rows]
    trainZ = Z[:train_rows]
    testX = X[train_rows:]
    testZ = Z[train_rows:]
    return trainX,trainZ,startprob_org,transmat_org,means_org,covars_org,testX,testZ

def training(trainX,trainZ,k,startprob_org,transmat_org,means_org,covars_org):
    model_p = GaussianHMM(n_components=k, covariance_type="diag", n_iter=1000).fit(trainX)
    #evaluating training
    hidden_states = model_p.predict(trainX)
    transmat=model_p.transmat_
    means=model_p.means_
    covars=model_p.covars_
    startprob=model_p.startprob_

    l = list(permutations(range(0, k)))
    acctrain=1000
    min_covars=0
    min_acc=1000

    for p in l:
        var_trainZ=trainZ
        for i in range(trainZ.shape[0]):
            var_trainZ[i]=p[trainZ[i]]
        min_acc=np.mean( hidden_states == var_trainZ )
        if(min_acc<acctrain):
            acctrain=min_acc
            perm_m=p

    # acctrain=norm(hidden_states-var_trainZ,2)

    # for p in l:
    #     var_means=means
    #     var_means[:,:]=var_means[p,:]
    #     if(min_means>norm(var_means-means_org,2)):
    #         min_means=norm(var_means-means_org,2)
    #         perm_m=p

    var_means = means
    var_means[:, :] = var_means[perm_m, :]
    min_means = norm(var_means - means_org, 2)

    var_start=startprob.reshape(k,1)
    var_start[:,:]=var_start[perm_m,:]
    min_start=norm(var_start-startprob_org.reshape(k,1),1)

    var_trans=transmat
    var_trans[:,:]=var_trans[perm_m,:]
    var_trans[:,:]=var_trans[:,perm_m]
    min_transmat=norm(var_trans-transmat_org,1)

    var_covars=covars
    var_covars[:,:,:]=var_covars[p,:,:]
    for i in range(k):
        min_covars=min_covars+norm(var_covars[i]-covars_org[i],2)
    min_covars=min_covars/k



    return min_start,min_transmat,min_means,min_covars,acctrain,model_p,perm_m


def testing(testX, testZ, k, model_p,perm_m):
    #model_p = GaussianHMM(n_components=k, covariance_type="diag", n_iter=1000).fit(trainX)
    # evaluating training
    predZ = model_p.predict(testX)
    var_testZ=testZ
    for i in range(testZ.shape[0]):
        var_testZ[i]=perm_m[testZ[i]]
    #acctest=norm(predZ-var_testZ,2)
    acctest = np.mean(predZ == var_testZ)

    return acctest



if __name__=="__main__":
    print "Setting experiment values"

    input=genfromtxt('experiment_values.csv', delimiter=',',dtype='int')
    input=input[1:,1:]
    outputarr = np.empty([51,12])


    for i in range(0,51):
        exp,n,k,l=input[i,:]
        trainX, trainZ,startprob_org,transmat_org,means_org,covars_org,testX, testZ=createSample(n,l,k)
        startTimeTrain = datetime.now()
        min_start, min_transmat, min_means, min_covars, acctrain, model_p, perm_m = training(trainX, trainZ, k,
                                                                                             startprob_org, transmat_org,
                                                                                             means_org, covars_org)
        endTimeTrain=datetime.now()
        timeTrain= endTimeTrain - startTimeTrain
        acctest=testing(testX, testZ,k,model_p,perm_m)

        timeTest=datetime.now()-endTimeTrain

        outputarr[i,:]= exp,timeTrain.total_seconds(),timeTest.total_seconds(),n,k,l,acctrain,acctest,min_start,min_transmat,min_means,min_covars


    np.savetxt('experimentall.out',outputarr, delimiter=',')




