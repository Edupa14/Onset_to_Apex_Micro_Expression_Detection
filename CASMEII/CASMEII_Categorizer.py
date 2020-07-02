import os
import numpy as np
import pandas as pd
import shutil



path='D:/University/Detecting Forced Emotions through Micro-Expression Recognition using Neural Networks/Datasets/CASMEII/CASME2_RAW_selected/CASME2_RAW_selected/'

catdatafile = pd.read_excel('../../../Datasets/CASMEII/cat_apex.xlsx')
catdata = np.array(catdatafile)


# print(catdata)
# print(namedata)


targetpath= '../../../Datasets/CASMEII_categorical_apex/'

directorylisting = os.listdir(path)


count=0
cats=[]
for item in catdata:
    if item[2] not in cats:
        cats.append(item[2])
        os.mkdir(targetpath+item[2], mode=0o777)

for subject in directorylisting:
    # print(subject)
    subjectdirectorylisting=os.listdir(path+subject)
    for video in subjectdirectorylisting:
        videopath = path+subject +'/'+ video
        found=False
        for vidid in range(len(catdata)):
            if catdata[vidid][1]==str(video) and ("sub"+str(catdata[vidid][0])==subject or "sub0"+str(catdata[vidid][0])==subject):
                print(video,catdata[vidid][2],catdata[vidid])
                # print(str(targetpath)+str(catdata[vidid][2])+"/"+str(catdata[vidid][0])+'_'+str(video))
                os.mkdir(targetpath + catdata[vidid][2]+"/"+str(catdata[vidid][0])+'_'+str(video), mode=0o777)
                viddirectorylisting = os.listdir(videopath)
                for image in viddirectorylisting:
                    # print(image)
                    if "img" + str(catdata[vidid][3]) + ".jpg" == str(image) :
                        print(vidid, image)
                        print(videopath + "/" + str(image))
                        shutil.copyfile(videopath + "/" + str(image),
                                        str(targetpath) + str(catdata[vidid][2]) + "/" + str(
                                            catdata[vidid][0]) + '_' + str(video) + '/1' + str(image))
                    if  "img"+str(catdata[vidid][4] )+".jpg"== str(image):
                        print(vidid,image)
                        print(videopath+"/"+str(image))
                        shutil.copyfile(videopath+"/"+str(image), str(targetpath)+str(catdata[vidid][2])+"/"+str(catdata[vidid][0])+'_'+str(video)+'/2'+str(image))
                if found==True:
                    print("multiple",video,catdata[vidid])
                found=True
                count+=1
        if found==False:
            print("not found",video)
        # print(str(subjectdirectorylisting)+str(video))
        continue

print(count)