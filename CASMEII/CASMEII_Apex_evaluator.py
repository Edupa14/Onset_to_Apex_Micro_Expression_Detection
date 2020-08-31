import pandas as pd
import numpy
import os

disgustpath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/disgust/'
fearpath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/fear/'
happinesspath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/happiness/'
otherspath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/others/'
repressionpath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/repression/'
sadnesspath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/sadness/'
surprisepath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer/surprise/'



paths=[disgustpath,  happinesspath,otherspath,repressionpath,surprisepath]
catdatafile = pd.read_excel('../../cat_apex.xlsx')
data = numpy.array(catdatafile)

count=0
for pi in range(len(paths)):
    directorylisting = os.listdir(paths[pi])
    print(pi)
    for video in range(len(directorylisting)):
        if directorylisting[video]!="4_EP12_01f":

            for item in data:
                print(str(item[0])+"_"+str(item[1]),directorylisting[video])
                if str(item[0])+"_"+str(item[1])==directorylisting[video]:
                    # Framefound=False
                    # first=None
                    # last=None
                    # print(framelistinglist[count])
                    # for frame in range(len(framelistinglist[count])):
                    #     if framelistinglist[count][frame]==item[4]-item[3]:
                    #         Framefound=True
                    #         first=frame-1
                    #     # elif Framefound==True:
                    #     #     last=frame-1
                    #         break
                    print(int(item[4]))
                    directorylistingvid = os.listdir(video)
                    print(directorylistingvid)
                    # print(first)
                    # print(framelistinglist[count][first-1:last+2],framelistinglist[count][first],framelistinglist[count][last])
                    # print(item[4]-item[3])
                    count+=1
                    break
