import os
import cv2
import dlib
import numpy
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd

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
    ans = numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    return ans


def annotate_landmarks(img, landmarks, font_scale=0.4):
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=font_scale,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img


path='../../../CAS(ME)2_categorical/'
negativepath =path+'negative/'
positivepath = path+'positive/'
surprisepath = path+'surprise/'



paths=[negativepath,positivepath, surprisepath]

segmentName = 'UpperFace'
sizeH=128
sizeV=128
sizeD=140

segment_training_list = []
counting = 0
framelistinglist=[]
for typepath in (paths):
    directorylisting = os.listdir(typepath)
    print(typepath)

    for video in directorylisting:
        if video!="4_EP12_01f":

            videopath = typepath + video
            segment_frames = []

            framelisting = os.listdir(videopath)
            if sizeD<=len(framelisting):
                val=int((len(framelisting)/2)-(sizeD/2))
                framerange = [x+val for x in range(sizeD)]
            else:
                tempD1=sizeD//len(framelisting)
                tempD2 = sizeD%len(framelisting)
                framerange = []
                # for y in range (len(framelisting)):
                #     framerange.extend([y for _ in range(tempD1)])
                #     if y<tempD2:
                #         framerange.append(y)
                framerange.extend([y for y in range(len(framelisting))])

                framerange.extend([-1 for _ in range(sizeD-len(framelisting))])
                # framerange.extend([y for y in range(tempD2)])
            print(framerange,len(framerange))
            framelistinglist.append(framerange)
            for frame in framerange:
                if frame==-1:
                    segment_image = [[0 for _ in range(sizeH)] for _ in range(sizeV)]
                    # print(len(segment_image),len(segment_image[0]))
                    # print(segment_image)
                    segment_frames.append(segment_image)
                else:
                    imagepath = videopath + "/" + framelisting[frame]
                    image = cv2.imread(imagepath)
                    landmarks = get_landmark(image)
                    if counting < 1:
                        img = annotate_landmarks(image, landmarks)
                        imgplot = plt.imshow(img)
                        plt.show()
                    numpylandmarks = numpy.asarray(landmarks)
                    up = min(numpylandmarks[18][1], numpylandmarks[19][1], numpylandmarks[23][1], numpylandmarks[24][1]) - 20
                    down = max(numpylandmarks[31][1], numpylandmarks[32][1], numpylandmarks[33][1], numpylandmarks[34][1],
                               numpylandmarks[35][1]) + 5
                    left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
                    right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])
                    segment_image = image[up:down, left:right]
                    if counting < 1:
                        img = annotate_landmarks(segment_image, landmarks)
                        imgplot = plt.imshow(img)
                        plt.show()
                        counting += 1
                    segment_image = cv2.resize(segment_image, (sizeH, sizeV), interpolation=cv2.INTER_AREA)
                    segment_image = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)

                    segment_frames.append(segment_image)


            segment_frames = numpy.asarray(segment_frames)
            segment_videoarray = numpy.rollaxis(numpy.rollaxis(segment_frames, 2, 0), 2, 0)
            # print('aaaa', segment_frames)
            # print('bbbb', segment_videoarray)
            # print('cccc',segment_image)
            segment_training_list.append(segment_videoarray)

segment_training_list = numpy.asarray(segment_training_list)

segment_trainingsamples = len(segment_training_list)

segment_traininglabels = []

catdatafile = pd.read_excel('../../../CAS(ME)^2code_final(Updated).xlsx')
data = numpy.array(catdatafile)

count=0
for pi in range(len(paths)):
    directorylisting = os.listdir(paths[pi])
    print(pi)
    for video in range(len(directorylisting)):
        if directorylisting[video]!="4_EP12_01f":

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
                    segment_traininglabels.append(int(item[4])-int(item[3]))
                    # print(first)
                    # print(framelistinglist[count][first-1:last+2],framelistinglist[count][first],framelistinglist[count][last])
                    # print(item[4]-item[3])
                    count+=1
                    break


#-----------------


segment_traininglabels_cat = numpy.zeros((segment_trainingsamples,), dtype=int)

count=0
for pi in range(len(paths)):
    directorylisting = os.listdir(paths[pi])
    print(pi)
    for video in range(len(directorylisting)):
        if directorylisting[video] != "4_EP12_01f":
            segment_traininglabels_cat[count] = pi
            count+=1


segment_traininglabels_cat = np_utils.to_categorical(segment_traininglabels_cat, len(paths))
#-----------------

segment_traininglabels = np_utils.to_categorical(segment_traininglabels, sizeD)

# print(segment_traininglabels)

segment_training_data = [segment_training_list, segment_traininglabels,segment_traininglabels_cat]
(segment_trainingframes, segment_traininglabels,segment_traininglabels_cat) = (segment_training_data[0], segment_training_data[1],segment_training_data[2])
segment_training_set = numpy.zeros((segment_trainingsamples, 1,sizeH, sizeV, sizeD))
for h in range(segment_trainingsamples):
    segment_training_set[h][0][:][:][:] = segment_trainingframes[h, :, :, :]

segment_training_set = segment_training_set.astype('float32')
segment_training_set -= numpy.mean(segment_training_set)
segment_training_set /= numpy.max(segment_training_set)
print(segment_traininglabels)
print(segment_traininglabels_cat)

numpy.save('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD), segment_training_set)
numpy.save('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD), segment_traininglabels)
numpy.save('numpy_training_datasets/{0}_labels_cat_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD), segment_traininglabels_cat)

"""
----------------------------
segments:
----------------------------


UpperFace
----------------------------
up = min(numpylandmarks[18][1], numpylandmarks[19][1], numpylandmarks[23][1], numpylandmarks[24][1]) - 20
down = max(numpylandmarks[31][1], numpylandmarks[32][1], numpylandmarks[33][1], numpylandmarks[34][1],
          numpylandmarks[35][1]) + 5
left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])


Eyes
----------------------------  
up = min(numpylandmarks[18][1], numpylandmarks[19][1], numpylandmarks[23][1], numpylandmarks[24][1]) - 20
down = max(numpylandmarks[36][1], numpylandmarks[39][1], numpylandmarks[40][1], numpylandmarks[41][1],numpylandmarks[42][1], numpylandmarks[47][1], numpylandmarks[46][1], numpylandmarks[45][1]) +10
left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])        


LeftEye
----------------------------  
up=min(numpylandmarks[17][1],numpylandmarks[18][1],numpylandmarks[19][1],numpylandmarks[20][1],numpylandmarks[21][1])-20
down = max(numpylandmarks[36][1], numpylandmarks[39][1], numpylandmarks[40][1], numpylandmarks[41][1]) +10
left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
right = max(numpylandmarks[39][0], numpylandmarks[21][0])+10


RightEye
----------------------------   
up = min(numpylandmarks[22][1], numpylandmarks[23][1], numpylandmarks[24][1], numpylandmarks[25][1],
        numpylandmarks[26][1]) - 20
down = max(numpylandmarks[42][1], numpylandmarks[47][1], numpylandmarks[46][1], numpylandmarks[45][1]) + 10
right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])
left = min(numpylandmarks[22][0], numpylandmarks[42][0])-10


Nose
----------------------------     
up = numpylandmarks[27][1] - 5
down = max(numpylandmarks[31][1], numpylandmarks[32][1], numpylandmarks[33][1], numpylandmarks[34][1], numpylandmarks[35][1]) + 5
left = numpylandmarks[31][0]
right = numpylandmarks[35][0] 
"""