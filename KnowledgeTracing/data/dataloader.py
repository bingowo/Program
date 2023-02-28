import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.utils.data as Data
from Constant import Constants as C
from data.readdata import DataReader
from data.DKTDataSet import DKTDataSet

def getLoaderSet(data_path):
    handle = DataReader(data_path ,C.MAX_STEP, C.NUM_OF_QUESTIONS)
    ques, ans, paths = handle.getData()
    ddata = DKTDataSet(ques, ans, paths)
    C.Data = ddata
    Loader = Data.DataLoader(ddata, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=C.WORKERS)
    return Loader

def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    if dataset == 'contest513':
        trainLoader = getLoaderSet(C.Dpath + '/contest513/contest513_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getLoaderSet(C.Dpath + '/contest513/contest513_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'static2011':
        trainLoader = getLoaderSet(C.Dpath + '/statics2011/static2011_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getLoaderSet(C.Dpath + '/statics2011/static2011_test.txt')
        testLoaders.append(testLoader)

    return trainLoaders, testLoaders