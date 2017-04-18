import sys
sys.path.append('/home/zlp/dev/caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import DataManager as DM
import utilities
import util
from os.path import splitext
from multiprocessing import Process, Queue

import scipy
from scipy import ndimage
import math

class VNet(object):
    params=None
    dataManagerTrain=None
    dataManagerTest=None

    def __init__(self,params):
        self.params=params
        caffe.set_device(self.params['ModelParams']['device'])
        caffe.set_mode_gpu()

    def prepareDataThread(self, dataQueue):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']
        pos_neg_ratio = 1.0

        # nr_iter_dataAug = nr_iter*batchsize
        nr_iter_dataAug = nr_iter
        np.random.seed()
        whichDataList = np.random.randint(len(self.dataManagerTrain.trainingdb), size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        for data_index in whichDataList:
            print data_index
            """ Get Item Infos
            """
            item = self.dataManagerTrain.trainingdb[data_index]
            fpath = item["fpath"]
            candidates_pos = np.array(item['candidates_pos'], np.float)
            candidates_neg = np.array(item['candidates_neg'], np.float)
            """ Load image infos, While in numpy, an array is indexed in the opposite order (z,y,x)
            """
            img, origin, spacing = util.load_itk_image(fpath)
            img = img.astype(np.float32, copy=False)
            """ Apply Hounsfield Unit Window  
            The Hounsfield unit values will be windowed in the range
            HU_WINDOW to exclude irrelevant organs and objects.
            """
            HU_WINDOW = [-1000., 400.]
            img = util.hounsfield_unit_window(img, HU_WINDOW)
            """ Map src_range value to dst_range value
            """
            img = util.normalizer(img, src_range=HU_WINDOW, dst_range=[0., 1.])
            """ Transform candidates from world coordinates to voxel coordinates
            voxel coordinates candicate ['coordZ', 'coordY', 'coordX']
            """
            candidates_pos = util.world_2_voxel(candidates_pos, origin, spacing)
            candidates_neg = util.world_2_voxel(candidates_neg, origin, spacing)
            print len(candidates_pos), len(candidates_neg)
            # util.ims_vis([img[0,:,:], img[:,0,:],img[:,:,0]])
            """ Data Augmentation for postive candidates
            Translation, Rotation and Multi-Scale
            """
            # Translation for each axis
            translations_z = [0] #[-1, 0, 1]
            translations_y = [-1, 0, 1] #[-2, -1, 0, 1, 2]
            translations_x = [-1, 0, 1] # [-2, -1, 0, 1, 2]
            translations_list = []
            for axis_z in translations_z:
                for axis_y in translations_y:
                    for axis_x in translations_x:
                        translations_list.append([axis_z, axis_y, axis_x])
            # Postive candidates augmentation
            pos_num = len(candidates_pos)
            for pos_index in xrange(pos_num):
                for trans in translations_list:
                    new_cand = np.array(candidates_pos[pos_index], copy=True)
                    new_cand += np.array(trans)
                    candidates_pos = np.vstack((candidates_pos, new_cand))
            # Rotation: rotation already implement in translation
            # rotation = [90, 180, 270]
            # Multi-Scale
            scales = [[8, 16, 16], [16, 28, 28], [24, 40, 40]]
            """ Prepare postive patch
            all patch zoom to fixed shape
            """
            fixed_shape = [24, 40, 40]
            # fig, axes = plt.subplots(1, 2)
            # plt.ion() # turn on interactive mode
            candidates_pos_patch = list()
            for pos_index in xrange(len(candidates_pos)):
                cand = candidates_pos[pos_index]
                # crop image patch based on cand and scales and zoom to fixed size
                for scale in scales:
                    patch_shape = np.array(scale)
                    # crop patch
                    patch_b = util.crop_patch(img.shape, patch_shape, cand)
                    patch_e = patch_b + patch_shape
                    im_patch = img[patch_b[0]:patch_e[0], patch_b[1]:patch_e[1], patch_b[2]:patch_e[2]]
                    # resize image
                    real_resize = np.array(fixed_shape, dtype=np.float) / im_patch.shape
                    im_patch = scipy.ndimage.interpolation.zoom(im_patch, real_resize)
                    # util.vis_seg(axes, [im_patch[im_patch.shape[0]//2,:,:], img[cand[0],:,:]])
                    candidates_pos_patch.append(im_patch)
            """construct potive candidates
            """
            batch_neg_num = int(math.ceil(batchsize / (float(pos_neg_ratio) + 1.0)))
            batch_pos_num = batchsize - batch_neg_num
            # candidates_pos_patch is a list
            if len(candidates_pos_patch) < batch_pos_num:
                candidates_pos_patch = candidates_pos_patch * (int(math.ceil(float(batch_pos_num)/len(candidates_pos_patch))))
            else:
                # shuffle
            candidates_pos_patch = candidates_pos_patch[0:batch_pos_num]
            print len(candidates_pos_patch)
            """construct negtive candidates
            """
            # candidates_neg is a array
            if len(candidates_neg) < batch_neg_num:
                neg_list = np.random.randint(len(candidates_neg), size=int(batch_neg_num-len(candidates_neg)))
                for neg_index in neg_list:
                    candidates_neg = np.vstack(candidates_neg, candidates_neg[neg_index])
            else:
                # shuffle
            candidates_neg = candidates_neg[0:batch_neg_num]
            candidates_neg_patch = list()
            for neg_index in xrange(len(candidates_neg)):
                cand = candidates_neg[neg_index]
                patch_shape = np.array(fixed_shape)
                # crop patch
                patch_b = util.crop_patch(img.shape, patch_shape, cand)
                patch_e = patch_b + patch_shape
                im_patch = img[patch_b[0]:patch_e[0], patch_b[1]:patch_e[1], patch_b[2]:patch_e[2]]
                candidates_neg_patch.append(im_patch)
            print len(candidates_neg_patch)


            



            
            exit()
            # crop candidates


        # for whichData,whichDataForMatching in zip(whichDataList,whichDataForMatchingList):
        #     filename, ext = splitext(keysIMG[whichData])

        #     currGtKey = filename + '_segmentation' + ext
        #     currImgKey = filename + ext

        #     # data agugumentation through hist matching across different examples...
        #     ImgKeyMatching = keysIMG[whichDataForMatching]

        #     defImg = numpyImages[currImgKey]
        #     defLab = numpyGT[currGtKey]

        #     defImg = utilities.hist_match(defImg, numpyImages[ImgKeyMatching])

        #     if(np.random.rand(1)[0]>0.5): #do not apply deformations always, just sometimes
        #         defImg, defLab = utilities.produceRandomlyDeformedImage(defImg, defLab,
        #                             self.params['ModelParams']['numcontrolpoints'],
        #                                        self.params['ModelParams']['sigma'])

        #     weightData = np.zeros_like(defLab,dtype=float)
        #     weightData[defLab == 1] = np.prod(defLab.shape) / np.sum((defLab==1).astype(dtype=np.float32))
        #     weightData[defLab == 0] = np.prod(defLab.shape) / np.sum((defLab == 0).astype(dtype=np.float32))

        #     dataQueue.put(tuple((defImg,defLab, weightData)))

    def trainThread(self,dataQueue,solver):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        batchData = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        #only used if you do weighted multinomial logistic regression
        batchWeight = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0],
                               self.params['DataManagerParams']['VolSize'][1],
                               self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        train_loss = np.zeros(nr_iter)
        for it in range(5):
            # for i in range(batchsize):
            #     [defImg, defLab, defWeight] = dataQueue.get()

            #     batchData[i, 0, :, :, :] = defImg.astype(dtype=np.float32)
            #     batchLabel[i, 0, :, :, :] = (defLab > 0.5).astype(dtype=np.float32)
            #     batchWeight[i, 0, :, :, :] = defWeight.astype(dtype=np.float32)

            # solver.net.blobs['data'].data[...] = batchData.astype(dtype=np.float32)
            # solver.net.blobs['label'].data[...] = batchLabel.astype(dtype=np.float32)
            # #solver.net.blobs['labelWeight'].data[...] = batchWeight.astype(dtype=np.float32)
            # #use only if you do softmax with loss


            # solver.step(1)  # this does the training
            # train_loss[it] = solver.net.blobs['loss'].data

            if (np.mod(it, 10) == 0):
                plt.clf()
                plt.plot(range(0, it), train_loss[0:it])
                plt.pause(0.00000001)


            matplotlib.pyplot.show()


    def train(self):
        print self.params['ModelParams']['dirTrain']

        #we define here a data manage object
        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData() #loads in sitk format

        howManyImages = len(self.dataManagerTrain.trainingdb)
        # howManyImages = len(self.dataManagerTrain.sitkImages)
        # howManyGT = len(self.dataManagerTrain.sitkGT)

        # assert howManyGT == howManyImages

        # print "The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT)

        test_interval = 50000
        # Write a temporary solver text file because pycaffe is stupid
        with open("solver.prototxt", 'w') as f:
            f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
            f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
            f.write("momentum: 0.99 \n")
            f.write("weight_decay: 0.0005 \n")
            f.write("lr_policy: \"step\" \n")
            f.write("stepsize: 20000 \n")
            f.write("gamma: 0.1 \n")
            f.write("display: 1 \n")
            f.write("snapshot: 500 \n")
            f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
            #f.write("test_iter: 3 \n")
            #f.write("test_interval: " + str(test_interval) + "\n")

        f.close()
        # solver = caffe.SGDSolver("solver.prototxt")
        # os.remove("solver.prototxt")

        # if (self.params['ModelParams']['snapshot'] > 0):
        #     solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(
        #         self.params['ModelParams']['snapshot']) + ".solverstate")

        # plt.ion()

        # numpyImages = self.dataManagerTrain.getNumpyImages()
        # numpyGT = self.dataManagerTrain.getNumpyGT()

        #numpyImages['Case00.mhd']
        #numpy images is a dictionary that you index in this way (with filenames)

        # for key in numpyImages:
        #     mean = np.mean(numpyImages[key][numpyImages[key]>0])
        #     std = np.std(numpyImages[key][numpyImages[key]>0])

        #     numpyImages[key]-=mean
        #     numpyImages[key]/=std

        dataQueue = Queue(30) #max 50 images in queue
        dataPreparation = [None] * self.params['ModelParams']['nProc']

        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=self.prepareDataThread, args=(dataQueue,))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        self.trainThread(dataQueue, solver=1)


    def test(self):
        self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        self.dataManagerTest.loadTestData()

        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        os.path.join(self.params['ModelParams']['dirSnapshots'],"_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

        numpyImages = self.dataManagerTest.getNumpyImages()

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])

            numpyImages[key] -= mean
            numpyImages[key] /= std

        results = dict()

        for key in numpyImages:

            btch = np.reshape(numpyImages[key],[1,1,numpyImages[key].shape[0],numpyImages[key].shape[1],numpyImages[key].shape[2]])

            net.blobs['data'].data[...] = btch

            out = net.forward()
            l = out["labelmap"]
            labelmap = np.squeeze(l[0,1,:,:,:])

            results[key] = np.squeeze(labelmap)

            self.dataManagerTest.writeResultsFromNumpyLabel(np.squeeze(labelmap),key)

