import numpy
import os

segmentName = 'CASMEII_UpperFace_SelectiveDivideAndConquer_NEW_mod_NEW_edit'
sizeH = 128
sizeV = 128
sizeD = 1


segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
print(segment_traininglabels)
cat=[0]*3
for item in segment_traininglabels:
    for c in range(len(cat)):
        if item[c]==1:
            cat[c]+=1

print(cat)




cat=[0]*7
dir=0
for typepath in os.listdir('../../../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod_NEW/'):
    directorylisting = os.listdir('../../../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod_NEW/'+typepath)
    for video in directorylisting:
        cat[dir]+=1
    dir+=1
print(cat)
print(sum(cat))