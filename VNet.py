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
import h5py
from medpy import metric
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

class VNet(object):
    # params=None
    # dataManagerTrain=None
    # dataManagerTest=None

    def __init__(self,params):
        self.params=params
        caffe.set_device(self.params['ModelParams']['device'])
        caffe.set_mode_gpu()

    def prepareDataThread(self, dataQueue):
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']
        #pos_neg_ratio = 0.334
        pos_neg_ratio = 1

        # nr_iter_dataAug = nr_iter*batchsize
        nr_iter_dataAug = nr_iter
        np.random.seed()
        whichDataList = np.random.randint(len(self.dataManagerTrain.trainingdb), size=int(math.ceil(float(nr_iter_dataAug)/self.params['ModelParams']['nProc'])))
        for ind_data in whichDataList:
            """ Get Item Infos
            """
            item = self.dataManagerTrain.trainingdb[ind_data]
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
            voxel coordinates candidate ['coordZ', 'coordY', 'coordX']
            """
            candidates_pos = util.world_2_voxel(candidates_pos, origin, spacing)
            candidates_neg = util.world_2_voxel(candidates_neg, origin, spacing)
            # print len(candidates_pos), len(candidates_neg)
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
                    im_patch = util.crop_patch(img, patch_shape, cand)
                    # resize image
                    real_resize = np.array(fixed_shape, dtype=np.float) / im_patch.shape
                    im_patch = scipy.ndimage.interpolation.zoom(im_patch, real_resize)
                    # util.ims_vis(axes, [im_patch[im_patch.shape[0]//2,:,:], img[cand[0],:,:]])
                    candidates_pos_patch.append(im_patch)
            """construct potive candidates
            """
            batch_neg_num = int(math.ceil(batchsize / (float(pos_neg_ratio) + 1.0)))
            batch_pos_num = batchsize - batch_neg_num
            # candidates_pos_patch is a list
            if len(candidates_pos_patch) < batch_pos_num:
                candidates_pos_patch = candidates_pos_patch * (int(math.ceil(float(batch_pos_num)/len(candidates_pos_patch))))
            else:
                # shuffle candidates_pos_patch list
                np.random.shuffle(candidates_pos_patch)
            candidates_pos_patch = candidates_pos_patch[0:batch_pos_num]
            # print len(candidates_pos_patch)
            """construct negtive candidates
            """
            # candidates_neg is a array
            if len(candidates_neg) < batch_neg_num:
                neg_list = np.random.randint(len(candidates_neg), size=int(batch_neg_num-len(candidates_neg)))
                for neg_index in neg_list:
                    candidates_neg = np.vstack((candidates_neg, candidates_neg[neg_index]))
            else:
                # shuffle candidates_neg array
                np.random.shuffle(candidates_neg)
            candidates_neg = candidates_neg[0:batch_neg_num]
            candidates_neg_patch = list()
            for neg_index in xrange(len(candidates_neg)):
                cand = candidates_neg[neg_index]
                patch_shape = np.array(fixed_shape)
                # crop patch
                im_patch = util.crop_patch(img, patch_shape, cand)
                candidates_neg_patch.append(im_patch)
            # print len(candidates_neg_patch)

            # Construct batch
            candidates_patch = list()
            candidates_patch.extend(candidates_pos_patch)
            candidates_patch.extend(candidates_neg_patch)
            batch_data = util.im_list_to_blob(candidates_patch)
            batch_label = np.zeros((batchsize, 1, 1, 1, 1), dtype=np.int8)
            batch_label[0:batch_pos_num,0,0,0,0] = 1
            # print label
            assert len(batch_data) == len(batch_label), 'Each sample must have corresponding label'
            perm = np.random.permutation(np.arange(len(batch_label)))
            batch_data = batch_data[perm]
            batch_label = batch_label[perm]
            # print batch_data.shape, batch_label.shape
            # print batch_label

            dataQueue.put(tuple((batch_data, batch_label)))

    def trainThread(self,dataQueue,solver):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        # batchData = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        # batchLabel = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        # #only used if you do weighted multinomial logistic regression
        # batchWeight = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0],
        #                        self.params['DataManagerParams']['VolSize'][1],
        #                        self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        train_loss = np.zeros(nr_iter)
        for it in range(nr_iter):
            [batch_data, batch_label] = dataQueue.get()
            solver.net.blobs['data'].data[...] = batch_data.astype(dtype=np.float32)
            solver.net.blobs['label'].data[...] = batch_label.astype(dtype=np.float32)
            # vis the data
            vflag = False
            if vflag:
                vis_num = 20
                fig, axes = plt.subplots(1, vis_num)
                vis_data = [batch_data[v,0,11,:,:] for v in xrange(vis_num)]
                vis_label = [batch_label[v,0,0,0,0] for v in xrange(vis_num)]
                print vis_label
                util.ims_vis(axes,vis_data)
                exit()

            solver.step(1)

            train_loss[it] = solver.net.blobs['loss'].data
            plot_interval = 100
            if (np.mod(it+1, plot_interval) == 0):
                plt.clf()
                # plot 0, 99, 199, 299
                x = np.array(range(0, it, plot_interval)) - 1
                x[0] = 0
                x = np.append(x,it)
                plt.plot(x, train_loss[x])
                plt.pause(0.00000001)
            matplotlib.pyplot.show()
            # save plot
            if (np.mod(it+1, plot_interval) == 0):
                plt.savefig(self.params['ModelParams']['SnapshotPrefix'] + '_plot.png')
        print 'solving done'

    def train(self):
        print self.params['ModelParams']['dirTrain']

        #we define here a data manage object
        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData() #loads in sitk format

        howManyImages = len(self.dataManagerTrain.trainingdb)
        print "The dataset has {} data".format(howManyImages)

        test_interval = 5000000
        # Write a temporary solver text file because pycaffe is stupid
        with open("solver.prototxt", 'w') as f:
            f.write('train_net: "{}" \n'.format(self.params['ModelParams']['prototxtTrain']))
            f.write('base_lr: {} \n'.format(self.params['ModelParams']['baseLR']))
            f.write('momentum: {} \n'.format(0.99))
            f.write('weight_decay: {} \n'.format(0.0005))
            f.write('lr_policy: "{}" \n'.format('step'))
            f.write('stepsize: {} \n'.format(self.params['ModelParams']['SnapshotIters']))
            f.write('gamma: {} \n'.format(0.1))
            f.write('display: {} \n'.format(100))
            f.write('snapshot: {} \n'.format(self.params['ModelParams']['SnapshotIters']))
            f.write('snapshot_prefix: "{}" \n'.format(self.params['ModelParams']['SnapshotPrefix']))
            #f.write("test_iter: 3 \n")
            #f.write("test_interval: " + str(test_interval) + "\n")

        f.close()
        solver = caffe.SGDSolver("solver.prototxt")
        os.remove("solver.prototxt")

        if (self.params['ModelParams']['snapshot'] > 0):
            print "solver restore ..."
            solver.restore(self.params['ModelParams']['SnapshotPrefix'] + "_iter_" + str(
                self.params['ModelParams']['snapshot']) + ".solverstate")

        plt.ion()

        dataQueue = Queue(30) #max 50 images in queue
        dataPreparation = [None] * self.params['ModelParams']['nProc']

        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=self.prepareDataThread, args=(dataQueue,))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        self.trainThread(dataQueue, solver)

    def prepareTestDataThread(self, dataQueue, index):
        whichDataList = range(index[0],index[1])
        for ind_data in whichDataList:
            """ Get Item Infos
            """
            item = self.dataManagerTest.testdb[ind_data]
            seriesuid = item['seriesuid']
            fpath = item["fpath"]
            candidates_pos_world = np.array(item['candidates_pos'], np.float)
            candidates_neg_world = np.array(item['candidates_neg'], np.float)
            # print ind_data, fpath
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
            voxel coordinates candidate ['coordZ', 'coordY', 'coordX']
            """
            num_pos = len(candidates_pos_world)
            num_neg = len(candidates_neg_world)

            if num_pos == 0:
                candidates_world = candidates_neg_world
            else:
                candidates_world = np.concatenate((candidates_pos_world, candidates_neg_world))
            candidates = util.world_2_voxel(candidates_world, origin, spacing)
            num_cand = len(candidates)
            # print num_pos, num_neg, num_cand
            """ Data Augmentation for postive candidates
            Translation, Rotation and Multi-Scale
            """
            # Multi-Scale
            scales = [[8, 16, 16], [16, 28, 28], [24, 40, 40]]
            num_scale = len(scales)
            """ Prepare postive patch
            all patch zoom to fixed shape
            """
            fixed_shape = [24, 40, 40]
            #fig, axes = plt.subplots(1, 2)
            # plt.ion() # turn on interactive mode
            candidates_patch = dict()
            for ind_scale in xrange(num_scale):
                candidates_patch['scale_{}'.format(ind_scale)] = list()

            for ind_cand in xrange(num_cand):
                cand = candidates[ind_cand]
                # crop image patch based on cand and scales and zoom to fixed size
                for ind_scale in xrange(num_scale):
                    patch_shape = np.array(scales[ind_scale])
                    # crop patch
                    im_patch = util.crop_patch(img, patch_shape, cand)
                    # resize image
                    real_resize = np.array(fixed_shape, dtype=np.float) / im_patch.shape
                    im_patch = scipy.ndimage.interpolation.zoom(im_patch, real_resize)
                    # util.ims_vis(axes, [im_patch[im_patch.shape[0]//2,:,:], img[cand[0],:,:]])
                    candidates_patch['scale_{}'.format(ind_scale)].append(im_patch)
            # label
            if num_pos == 0:
                label = np.zeros(num_neg,dtype=np.float)
            else:
                label_pos = np.zeros(num_pos, dtype=np.float) + 1
                label_neg = np.zeros(num_neg,dtype=np.float)
                label = np.concatenate((label_pos, label_neg))

            # Construct batch
            blobs = dict()
            for ind_scale in xrange(num_scale):
                key = 'scale_{}'.format(ind_scale)
                blobs[key] = util.im_list_to_blob(candidates_patch[key])

            dataQueue.put(tuple((seriesuid, candidates_world, blobs, label)))


    def testThread(self,dataQueue,net):
        # store the results, key is the seriesuid
        results = {}
        # all seriesuids
        seriesuids_list = list()
        for ind_data in xrange(len(self.dataManagerTest.testdb)):
            item = self.dataManagerTest.testdb[ind_data]
            seriesuid = item['seriesuid']
            fpath = item["fpath"]
            seriesuids_list.append(seriesuid)
        # forward
        # probability = np.zeros((num_cand, num_scale), dtype=np.float)
        max_batch_size = 64
        while len(seriesuids_list) > 0:
            # for each seriesuid
            print dataQueue.qsize()
            [seriesuid, candidates_world, blobs, label] = dataQueue.get()
            assert seriesuid in seriesuids_list, 'error seriesuid duplicated' 
            print dataQueue.qsize()
            # remove seriesuid
            seriesuids_list.remove(seriesuid)
            # print infos
            print '---{}/{}---'.format(len(self.dataManagerTest.testdb)-len(seriesuids_list), len(self.dataManagerTest.testdb))
            print '{}'.format(seriesuid)

            num_cand, num_scale = len(label), len(blobs)
            probability = np.zeros((num_cand, num_scale), dtype=np.float)
            for ind_scale in xrange(num_scale):
                key = 'scale_{}'.format(ind_scale)
                blob_data = blobs[key]
                print key
                # Prevent batch size too large to exceed gpu memory
                for split in xrange(int(np.ceil(float(num_cand)/max_batch_size))):
                # for split in xrange(num_cand // max_batch_size + 1):
                    batch_start =  max_batch_size * split
                    batch_end = max_batch_size * (split + 1 )
                    if batch_end > num_cand:
                        batch_end = num_cand
                    input_data = blob_data[batch_start:batch_end,:,:,:,:]
                    print batch_start, batch_end, input_data.shape
                    net.blobs['data'].reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],input_data.shape[3],input_data.shape[4])
                    print net.blobs['data'].data[...].shape
                    # net.reshape() # optinal -- the net will reshape automatically before a call to forward()
                    net.blobs['data'].data[...] = input_data.astype(dtype=np.float32)
                    output_data = net.forward()
                    prob = output_data["classifier"]
                    print prob.shape
                    # print prob.shape
                    probability[batch_start:batch_end, ind_scale] = prob[:,1]

            evaluation_array = np.zeros((num_cand, 7), dtype=np.float)
            for ind_cand in xrange(num_cand):
                # coordZ,coordY,coordX to coordX,coordY,coordZ
                evaluation_array[ind_cand, 0:3] = candidates_world[ind_cand,::-1]
                evaluation_array[ind_cand, 3] = label[ind_cand]
                evaluation_array[ind_cand, 4:7] = probability[ind_cand, :]
            results[seriesuid] = evaluation_array
            print '--- done ---'

        fsave = os.path.join(self.params['ModelParams']['dirResult'], 'Candidates_V2.h5')
        h5f = h5py.File(fsave, 'w')
        for k,v in results.items():
            h5f.create_dataset(k, data=v)
        h5f.close()
        print 'Results saved at {}'.format(fsave)
        print '--- DONE ---'
        return fsave


    def test(self):
        self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        self.dataManagerTest.loadTestData()

        # self.dataManagerTest.testdb = self.dataManagerTest.testdb[0:6]
        howManyImages = len(self.dataManagerTest.testdb)
        print "The dataset has {} data".format(howManyImages)

        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        self.params['ModelParams']['SnapshotPrefix'] + "_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel",
                        caffe.TEST)

        dataQueue = Queue(10) #max 50 images in queue
        dataPreparation = [None] * self.params['ModelParams']['nProc']
        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            # split data
            split_start = int(np.ceil(float(howManyImages)/self.params['ModelParams']['nProc'])) * proc
            split_end = int(np.ceil(float(howManyImages)/self.params['ModelParams']['nProc'])) * (proc + 1)
            if split_end > howManyImages:
                split_end = howManyImages
            index = (split_start, split_end)

            dataPreparation[proc] = Process(target=self.prepareTestDataThread, args=(dataQueue,index))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        fsave = self.testThread(dataQueue, net)

        ''' Evaluation
        '''
        output_dir = os.path.join(self.params['ModelParams']['dirResult'], 'evaluation')
        self.eval(fsave,output_dir)
        print 'Evaluation Done'

    def eval(self, eval_filename=None, output_dir=None):
        assert eval_filename is not None, 'eval_filename is not assigned'
        assert output_dir is not None, 'output_dir is not assigned'

        annotations_filename          = os.path.join(self.params['ModelParams']['dirEvaluation'],'annotations/annotations.csv')
        annotations_excluded_filename = os.path.join(self.params['ModelParams']['dirEvaluation'],'annotations/annotations_excluded.csv')
        seriesuids_filename           = os.path.join(self.params['ModelParams']['dirEvaluation'],'annotations/seriesuids.csv')
        results_filename              = eval_filename
        outputDir                     = output_dir
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        assert os.path.exists(results_filename), 'File does not exist: {}'.format(results_filename)

        h5f = h5py.File(results_filename, 'r')
        # np.set_printoptions(precision=1)
        seriesuids_list = list()
        evaluation_array = np.zeros((1, 7), dtype=np.float)
        for k,v in h5f.items():
            # seriesuid.append(k)
            seriesuid = list()
            seriesuid.append(k)
            seriesuid = seriesuid*(v[:].shape[0])
            seriesuids_list.extend(seriesuid)
            evaluation_array = np.concatenate((evaluation_array, v[:]), axis=0)
        h5f.close()
        evaluation_array = evaluation_array[1:]
        # print evaluation_array.shape, len(seriesuids_list)
        # print evaluation_array[0]
        # print seriesuids_list[0]
        label = evaluation_array[:,3]
        prob = evaluation_array[:,4:7]
        # average scores
        ave_scores = np.mean(prob, axis=1)
        # weight = np.array([.3,.5,.2]).reshape(1,3)
        # ave_scores = np.sum(prob*weight, axis=1)
        # p = ave_scores[(label<0.5) & (ave_scores>0.8)]
        # print p.shape
        # print p
        print metric.binary.jc(ave_scores>=0.5, label >=0.5)
        print metric.binary.recall(ave_scores>=0.5, label >=0.5)
        print metric.binary.precision(ave_scores>=0.5, label >=0.5)
        # ave_scores_list = ave_scores.tolist()
        # print len(ave_scores_list)
        # print ave_scores_list[0]
        #seriesuid,coordX,coordY,coordZ,probability
        results = list()
        results.append(['seriesuid','coordX', 'coordY', 'coordZ', 'probability'])
        for ind_seriesuid in xrange(len(seriesuids_list)):
            seriesuid = seriesuids_list[ind_seriesuid]
            coordX = evaluation_array[ind_seriesuid, 0]
            coordY = evaluation_array[ind_seriesuid, 1]
            coordZ = evaluation_array[ind_seriesuid, 2]
            probability = ave_scores[ind_seriesuid]
            results.append([seriesuid, coordX, coordY, coordZ, probability])

        results_filename = os.path.splitext(results_filename)[0] + '.csv'
        util.write_csv(results_filename, results)
        ### noduleCADEvaluation ###
        noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir)
        print "Finished!"