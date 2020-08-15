import numpy
import os
from keras.utils import np_utils
import pandas as pd

segmentName = 'UpperFace'
sizeH=32
sizeV=32
sizeD=140
catdatafile = pd.read_excel('../../at_apex.csv')
data = numpy.array(catdatafile)
segment_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))

segment_traininglabels = numpy.zeros((len(segment_training_set),), dtype=int)
disgustpath = '../../CASMEII_categorical/disgust/'
# fearpath = '../../CASMEII_categorical/fear/'
happinesspath = '../../CASMEII_categorical/happiness/'
otherspath = '../../CASMEII_categorical/others/'
repressionpath = '../../CASMEII_categorical/repression/'
# sadnesspath = '../../CASMEII_categorical/sadness/'
surprisepath = '../../CASMEII_categorical/surprise/'
paths=[disgustpath,  happinesspath,otherspath,repressionpath,surprisepath]
count=0
for pi in range(len(paths)):
    directorylisting = os.listdir(paths[pi])
    print(pi)
    for video in range(len(directorylisting)):
        for item in data:
            if item[0]+"_"+item[1]==video:
                segment_traininglabels[count] = item[4]
                count+=1

segment_traininglabels = np_utils.to_categorical(segment_traininglabels, len(paths))
numpy.save('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD), segment_traininglabels)
