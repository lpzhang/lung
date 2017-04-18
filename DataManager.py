import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
import csv

class DataManager(object):
    params=None
    srcFolder=None
    resultsDir=None

    fileList=None
    gtList=None

    sitkImages=None
    sitkGT=None
    meanIntensityTrain = None

    def __init__(self,srcFolder,resultsDir,parameters):
        self.params=parameters
        self.srcFolder=srcFolder
        self.resultsDir=resultsDir
        self.candidates_file = join(srcFolder, "CSVFILES/candidates.csv")

    def load_candidates(self):
        self.candidates=list()
        with open(self.candidates_file, 'rb') as f:
            scvreader = csv.reader(f)
            for line in scvreader:
                self.candidates.append(line)

    def load_images(self):
        self.filedict=dict()
        for i in xrange(0,10):
            subset_dir = join(self.srcFolder, 'subset{}'.format(i))
            for f in listdir(subset_dir):
                fpath = join(subset_dir, f)
                if isfile(fpath) and 'mhd' in f:
                    self.filedict[splitext(f)[0]] = fpath

    def createImageFileList(self):
        '''IMDB store a list of items dict
        Each item dict has key:
        seriesuid, fpath, candidates
        candidates is a list of candidate which is ['coordX', 'coordY', 'coordZ', 'class']
        '''
        self.imdb = list()
        # new item start
        item = dict()
        seriesuid = self.candidates[1][0]
        item['seriesuid'] = seriesuid
        item['fpath'] = self.filedict[seriesuid]
        item['candidates_pos'] = list()
        item['candidates_neg'] = list()
        # for each candidate
        for i in xrange(1, len(self.candidates)):
            cand = self.candidates[i]
            if cand[0] != seriesuid:
                # save previous item
                self.imdb.append(item)
                # new item start
                item = dict()
                seriesuid = cand[0]
                item['seriesuid'] = seriesuid
                item['fpath'] = self.filedict[seriesuid]
                item['candidates_pos'] = list()
                item['candidates_neg'] = list()
            # append candicate ['coordX', 'coordY', 'coordZ', 'class'] to current item's candidates list
            # item['candidates'].append(cand[1:])
            # append candicate ['coordZ', 'coordY', 'coordX', 'class'] to current item's candidates list
            candidates_coord = [cand[3], cand[2], cand[1]]
            if int(cand[4]) > 0:
                item['candidates_pos'].append(candidates_coord)
            else:
                item['candidates_neg'].append(candidates_coord)
            # save last item
            if i == (len(self.candidates)-1):
                self.imdb.append(item)

    def loadTrainingData(self):
        self.load_candidates()
        self.load_images()
        self.createImageFileList()

        self.trainingdb = list()
        validation_set = 'subset9'
        # print len(self.imdb)
        for item in self.imdb:
            if validation_set in item['fpath']:
                continue
            elif len(item['candidates_pos']) == 0:
                continue
            else:
                self.trainingdb.append(item)

        # print len(self.trainingdb)



# a = list()
# b = np.array([1,2,3])
# a.append(b)
# b = np.array([11,22,33])
# a.append(b)
# b = np.array([111,222,333])
# a.append(b)
# b = np.array([1111,2222,3333])
# a.append(b)
# b = np.array([1111,2222,3333])
# a.append(b)

# print a
# c = np.array(a)
# print c

# print len(c)
# print c[0:3]