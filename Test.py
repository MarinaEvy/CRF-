
import math
import pandas as pd
import numpy as np
import RTLearner as rt
import BagLearner as bl
import LinRegLearner as lr


data=pd.read_csv('C:\Users\mjohnson6\Desktop\winequality-red.csv')
data=np.array(data)

train_rows = math.floor(0.6* data.shape[0])
test_rows = data.shape[0] - train_rows


# separate out training and testing data
trainX = data[:train_rows,0:-1]
trainY = data[:train_rows,-1]
testX = data[train_rows:,0:-1]
testY = data[train_rows:,-1]

learner_RDL = rt.RTLearner(leaf_size=50, verbose = False) # create a LinRegLearner
learner_RDL.addEvidence(trainX, trainY) # train it

predY_RDL = learner_RDL.query(trainX) # get the predictions
rmse_RDL = math.sqrt(((trainY - predY_RDL) ** 2).sum()/trainY.shape[0])
c_RDL = np.corrcoef(predY_RDL, y=trainY)

predY_RDL_Test=learner_RDL.query(testX) 
rmse_RDL_Test = math.sqrt(((testY - predY_RDL_Test) ** 2).sum()/testY.shape[0])
c_RDL_Test = np.corrcoef(predY_RDL_Test, y=testY)


learner_BL=bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
learner_BL.addEvidence(trainX, trainY)

predY_BL = learner_BL.query(trainX) # get the predictions
rmse_BL = math.sqrt(((trainY - predY_BL) ** 2).sum()/trainY.shape[0])
c_BL = np.corrcoef(predY_BL, y=trainY)

predY_BL_Test=learner_BL.query(testX) 
rmse_BL_Test = math.sqrt(((testY - predY_BL_Test) ** 2).sum()/testY.shape[0])
c_BL_Test = np.corrcoef(predY_BL_Test, y=testY)


learner_LR=bl.BagLearner(learner = lr.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
learner_LR.addEvidence(trainX, trainY)

predY_LR = learner_LR.query(trainX) # get the predictions
rmse_LR = math.sqrt(((trainY - predY_LR) ** 2).sum()/trainY.shape[0])
c_LR = np.corrcoef(predY_LR, y=trainY)

predY_LR_Test=learner_LR.query(testX) 
rmse_LR_Test = math.sqrt(((testY - predY_LR_Test) ** 2).sum()/testY.shape[0])
c_LR_Test = np.corrcoef(predY_LR_Test, y=testY)



print
print "----------------------------------------"
print "In sample results_RTL"
print "RMSE: ", rmse_RDL
print "Correlation: ", c_RDL[0,1], 
print
print "out of sample results_RTL"
print "RMSE: ", rmse_RDL_Test
print "Correlation: ", c_RDL_Test[0,1], 
print
print "----------------------------------------"
print "In sample results_Bagging with DT"
print "RMSE: ", rmse_RDL
print "Correlation: ", c_RDL[0,1], 
print
print "out of sample results_Bagging with DT"
print "RMSE: ", rmse_RDL_Test
print "Correlation: ", c_RDL_Test[0,1], 
print
print "----------------------------------------"
print "In sample results_Bagging with Regression"
print "RMSE: ", rmse_LR
print "Correlation: ", c_LR[0,1], 
print
print "out of sample results_Bagging with Regression"
print "RMSE: ", rmse_LR_Test
print "Correlation: ", c_LR_Test[0,1], 
