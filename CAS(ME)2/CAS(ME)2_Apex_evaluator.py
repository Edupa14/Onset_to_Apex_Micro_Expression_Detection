import pandas as pd
import numpy
import os
import math
import statistics as stat
catdatafile = pd.read_excel('../../CAS(ME)^2code_final(Updated).xlsx')
data = numpy.array(catdatafile)
namedatafile = pd.read_excel('../../name.xlsx')
namedata = numpy.array(namedatafile)
count=0
for item in data:
    for item2 in namedata:
        # print(str(item2[0]),"s"+str(item[0]))
        if str(item2[1])==str(item[0]):
            # print(item2[0])
            item[0]=item2[0]
            break
path='../../CAS(ME)2_categorical_apex_selective/'
negativepath = path+'negative/'
positivepath = path+'positive/'
surprisepath = path+'surprise/'
othersepath = path+'others/'


paths=[negativepath,positivepath, surprisepath]
diffs=[]
count=0
for pi in range(len(paths)):
    directorylisting = os.listdir(paths[pi])
    print(pi)
    for video in range(len(directorylisting)):
        # if directorylisting[video]!="4_EP12_01f":
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
