import sys
import os
import numpy as np
import VNet as VN

basePath=os.getcwd()

params = dict()
params['DataManagerParams']=dict()
params['ModelParams']=dict()

#params of the algorithm
# params['ModelParams']['numcontrolpoints']=2
# params['ModelParams']['sigma']=15
params['ModelParams']['device']=0
params['ModelParams']['prototxtTrain']=os.path.join(basePath,'Prototxt/Net3DBN/train.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(basePath,'Prototxt/Net3DBN/test.prototxt')
params['ModelParams']['snapshot']=24000
params['ModelParams']['dirTrain']=os.path.join(basePath,'data/lungnodule')
params['ModelParams']['dirTest']=os.path.join(basePath,'data/lungnodule')
params['ModelParams']['dirEvaluation']=os.path.join(basePath,'evaluationScript')
params['ModelParams']['dirResult']=os.path.join(basePath,'result/Net3DBN') #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(basePath,'output/Net3DBN') #where to save the models while training
params['ModelParams']['SnapshotPrefix']= os.path.join(params['ModelParams']['dirSnapshots'],'Net3DBN_384')
params['ModelParams']['SnapshotIters'] = 8000
params['ModelParams']['batchsize'] = 384 #the batchsize
params['ModelParams']['numIterations'] = 24000 #the number of iterations
params['ModelParams']['baseLR'] = 0.01 #the learning rate, initial one
params['ModelParams']['nProc'] = 10 #the number of threads to do data augmentation



#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1,1,1.5],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([128,128,64],dtype=int)
params['DataManagerParams']['normDir'] = False #if rotates the volume according to its transformation in the mhd file. Not reccommended.

if not os.path.exists(params['ModelParams']['dirSnapshots']):
	os.makedirs(params['ModelParams']['dirSnapshots'])
if not os.path.exists(params['ModelParams']['dirResult']):
	os.makedirs(params['ModelParams']['dirResult'])

model=VN.VNet(params)
fsave = os.path.join(params['ModelParams']['dirResult'], 'Candidates_V2.h5')
output_dir = os.path.join(params['ModelParams']['dirResult'], 'evaluation_2')
model.eval(fsave,output_dir)
# model.test()
# model.train()
# train = [i for i, j in enumerate(sys.argv) if j == '-train']
# if len(train)>0:
#     model.train()

# test = [i for i, j in enumerate(sys.argv) if j == '-test']
# if len(test) > 0:
#     model.test()

