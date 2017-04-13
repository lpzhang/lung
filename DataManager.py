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
        item['candidates'] = list()
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
                item['candidates'] = list()
            # append candicate ['coordX', 'coordY', 'coordZ', 'class'] to current item's candidates list
            # item['candidates'].append(cand[1:])
            # append candicate ['coordZ', 'coordY', 'coordX', 'class'] to current item's candidates list
            item['candidates'].append([cand[3], cand[2], cand[1], cand[4]])
        self.imdb.append(item)
        self.imdb.append(item)

    def loadTrainingData(self):
        self.load_candidates()
        self.load_images()
        self.createImageFileList()

        self.trainingdb = list()
        validation_set = 'subset9'
        print len(self.imdb)
        for item in self.imdb:
            if validation_set in item['fpath']:
                continue
            else:
                self.trainingdb.append(item)

        print len(self.trainingdb)


# srcFolder = "/home/zlp/dev/lungnodule/data/lungnodule"
# d = DataManager(srcFolder,srcFolder,srcFolder)
# d.loadTrainingData()

a = ['56.08','67.85','311.92','0.5']
b = np.array(a, np.float)
print np.round(b).astype(np.int)
