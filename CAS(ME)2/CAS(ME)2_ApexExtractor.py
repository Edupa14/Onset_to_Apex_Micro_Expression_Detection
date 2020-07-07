import os
import numpy as np
import cv2
import dlib
from keras import backend as K
import shutil

K.set_image_dim_ordering('th')

# DLib Face Detection path setup
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmark(img):
    rects = detector(img, 1)
    if len(rects) > 1:
        pass
    if len(rects) == 0:
        pass
    ans = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    return ans


def annotate_landmarks(img, landmarks, font_scale=0.4):
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=font_scale,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img

path='../../CAS(ME)2_categorical/'

# catdatafile = pd.read_excel('../../../Datasets/CASMEII/cat_apex.xlsx')
# catdata = np.array(catdatafile)


# print(catdata)
# print(namedata)


targetpath= '../../CAS(ME)2_categorical_apex/'
if os.path.exists(targetpath ):
    shutil.rmtree(targetpath )
os.mkdir(targetpath , mode=0o777)
directorylisting = os.listdir(path)


count=0
cats=[]
for item in directorylisting:
    if item not in cats:
        cats.append(item)
        if os.path.exists(targetpath+item):
            shutil.rmtree(targetpath+item)
        os.mkdir(targetpath+item, mode=0o777)

for subject in directorylisting:
    # print(subject)
    subjectdirectorylisting=os.listdir(path+subject)
    for video in subjectdirectorylisting:
        videopath = path+subject +'/'+ video
        newvideopath=targetpath+subject +'/'+ video
        found=False
        # for vidid in range(len(catdata)):
            # if catdata[vidid][1]==str(video) and ("sub"+str(catdata[vidid][0])==subject or "sub0"+str(catdata[vidid][0])==subject):
            #     print(video,catdata[vidid][2],catdata[vidid])
                # print(str(targetpath)+str(catdata[vidid][2])+"/"+str(catdata[vidid][0])+'_'+str(video))
        if os.path.exists(newvideopath):
            shutil.rmtree(newvideopath)
        os.mkdir(newvideopath, mode=0o777)
        viddirectorylisting = os.listdir(videopath)
        print(videopath,viddirectorylisting)
        image = cv2.imread(videopath+'/'+viddirectorylisting[0])
        landmarks = get_landmark(image)
        numpylandmarks = np.asarray(landmarks)
        total_pos=[1]*len(numpylandmarks)
        img_pos=[]
        count=0
        for image in viddirectorylisting:
            img_pos.append([])
            image = cv2.imread(videopath+'/'+image)
            landmarks = get_landmark(image)
            numpylandmarks = np.asarray(landmarks)
            count+=1
            for pos in range(len(total_pos)):
                total_pos[pos]+=numpylandmarks[pos]
                img_pos[count-1].append(numpylandmarks[pos])
        avg_pos=[]
        for pos in total_pos:
            avg_pos.append(pos/count)
        max_diff=0
        max_diff_image=None
        count2=0
        for image in viddirectorylisting:
            diff=[]
            for pos in range(len(avg_pos)):
                diff.append(abs(avg_pos[pos]-img_pos[count2][pos]))
            count2+=1
            # print(diff,sum(diff),len(diff),sum(sum(diff)/len(diff)))
            avg_diff=sum(sum(diff)/len(diff))
            # print(max_diff,avg_diff)
            if max_diff<avg_diff:
                max_diff=avg_diff
                max_diff_image=image
        print(max_diff_image,max_diff)
        for image in viddirectorylisting:
            if viddirectorylisting[0] == image :
                # print(vidid, image,1)
                print(videopath + "/" + str(image))
                shutil.copyfile(videopath + "/" + str(image),
                                newvideopath+ '/1' + str(image))
            if  max_diff_image == image:
                # print(vidid,image,2)
                print(videopath+"/"+str(image))
                shutil.copyfile(videopath+"/"+str(image), newvideopath+'/2'+str(image))

        #         if found==True:
        #             print("multiple",video,catdata[vidid])
        #         found=True
        #         count+=1
        # if found==False:
        #     print("not found",video)
        # # print(str(subjectdirectorylisting)+str(video))
        # continue

