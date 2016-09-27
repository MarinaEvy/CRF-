"""
A simple wrapper for Bagging
"""

import numpy as np
import RTLearner as rt
import LinRegLearner as lr



class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose): #initilize the instance
        self.verbose=verbose
        self.kwargs=kwargs
        self.bags=bags
        self.boost=boost
        self.learner=learner
        pass 
        
           
    def addEvidence(self, X, Y): #train the dataset
        self.learners = [] #create an emoty array where we will store the learners
        for i in range(0,self.bags):
            self.learners.append(self.learner(**self.kwargs)) # create the learners (as many as the # of bags)
        if self.learner==rt.RTLearner: #if learner is random decision tree then train the bag learner using Random Tree     
            if self.boost==False:
                for l in self.learners:
                    a=np.random.randint(0, high=X.shape[0], size=X.shape[0])
                    dataX=X[a]
                    dataY=Y[a]
                    l.addEvidence(dataX, dataY)   
        if self.learner==lr.LinRegLearner: #if learner is regression then train the bag learner using regression
            if self.boost==False:
                for l in self.learners:
                    a=np.random.randint(0, high=X.shape[0], size=X.shape[0])
                    dataX=X[a]
                    dataY=Y[a]
                    l.addEvidence(dataX, dataY)
        if self.verbose:
            print self.learners
        return self.learners


    
    def query(self, points): #query the Y values by scnanning the TREE and then average the Y's
        if self.learner==rt.RTLearner: #query the points from the tree array in random trees
            Result=[]
            for l in self.learners:
                Result.append(l.query(points))             
        if self.learner==lr.LinRegLearner: #query the points by using the linear equation in regression
            Result=[]
            for l in self.learners:
                Result.append(l.query(points))     
        if self.verbose:
            print np.mean(Result, axis=0)
        return np.mean(Result, axis=0) #average the Y's of each bag. 
            
    