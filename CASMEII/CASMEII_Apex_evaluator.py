import pandas as pd
import numpy
import os
import math
import statistics as stat
disgustpath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/disgust/'
fearpath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/fear/'
happinesspath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/happiness/'
otherspath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/others/'
repressionpath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/repression/'
sadnesspath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/sadness/'
surprisepath = '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/surprise/'



paths=[disgustpath, fearpath, happinesspath,sadnesspath,otherspath,repressionpath,surprisepath]
catdatafile = pd.read_excel('../../cat_apex.xlsx')
data = numpy.array(catdatafile)
diffs=[]
count=0
for pi in range(len(paths)):
    directorylisting = os.listdir(paths[pi])
    print(pi)
    for video in range(len(directorylisting)):
        if directorylisting[video]!="4_EP12_01f":
            print(directorylisting[video])
            for item in data:
                # print(str(item[0])+"_"+str(item[1]),directorylisting[video])
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
                    directorylistingvid = os.listdir(paths[pi]+directorylisting[video])
                    for pic in directorylistingvid:
                        if pic[0]=='2':
                            print(pic[4:-4])
                            diffs.append(abs(item[4]-int(pic[4:-4])))
                    # print(directorylistingvid)
                    # print(first)
                    # print(framelistinglist[count][first-1:last+2],framelistinglist[count][first],framelistinglist[count][last])
                    # print(item[4]-item[3])
                    count+=1
                    break
print("MAE: ",(sum(diffs))/len(diffs))
print("SD: ",stat.stdev(diffs))
print("SE: ",stat.stdev(diffs)/(math.sqrt(len(diffs))))
