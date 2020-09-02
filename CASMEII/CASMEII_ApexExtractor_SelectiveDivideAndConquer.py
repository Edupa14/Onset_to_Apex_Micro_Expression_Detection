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


def generate_pos(imgs,landmark_list):
    image = cv2.imread(videopath + '/' + imgs[0])
    landmarks = get_landmark(image)
    numpylandmarks = np.asarray(landmarks)
    landmarksNos = len(numpylandmarks)
    total_pos = [1] * landmarksNos
    img_pos = []
    count = 0

    for image in viddirectorylisting:
        img_pos.append([])
        image = cv2.imread(videopath + '/' + image)
        landmarks = get_landmark(image)
        numpylandmarks = np.asarray(landmarks)
        count += 1
        img_pos[count - 1] = [1] * landmarksNos
        for pos in range(landmarksNos):
            # total_pos[pos] += (numpylandmarks[pos] - numpylandmarks[8])
            # print(numpylandmarks[pos], numpylandmarks[8], numpylandmarks[pos] - numpylandmarks[8])
            img_pos[count - 1][pos] = (numpylandmarks[pos] - numpylandmarks[8])
    return img_pos
def new_find_max(start,end,img_pos,vidlist,landmark_list):
    total_pos=[np.zeros(shape=2,dtype=int) for _ in range(len(img_pos[0]))]
    for imgval in img_pos:
        for posi in range(len(imgval)):
            # print(temp_tot[posi])
            total_pos[posi]+=imgval[posi]
            # print('a',temp_tot[posi],temp_tot)
    avg_pos = []
    for pos in total_pos:
        avg_pos.append(pos / count)
    # print(avg_pos[0],img_pos[0][0],avg_pos[0]-img_pos[0][0])
    max_diff = 0
    max_diff_image = None
    count2 = 0
    for image in range(start,end+1,1):
        diff = []
        # print(landmark_list,len(avg_pos))
        for pos in landmark_list:
            diff.append(abs(avg_pos[pos] - img_pos[count2][pos]))
        count2 += 1
        # print(sum(diff),len(diff),sum(sum(diff)/len(diff)))
        avg_diff = abs(sum(sum(diff) / len(diff)))
        # print(max_diff,avg_diff)
        if max_diff < avg_diff:
            max_diff = avg_diff
            max_diff_image = image
    return max_diff,vidlist[max_diff_image]



# def find_max(imgs):
#     image = cv2.imread(videopath + '/' + imgs[0])
#     landmarks = get_landmark(image)
#     numpylandmarks = np.asarray(landmarks)
#     landmarksNos = len(numpylandmarks)
#     total_pos = [1] * landmarksNos
#     img_pos = []
#     count = 0
#     landmark_list = [x for x in range(17, 27)]
#     landmark_list.extend([x for x in range(36, 68)])
#     for image in viddirectorylisting:
#         img_pos.append([])
#         image = cv2.imread(videopath + '/' + image)
#         landmarks = get_landmark(image)
#         numpylandmarks = np.asarray(landmarks)
#         count += 1
#         img_pos[count - 1] = [1] * landmarksNos
#         for pos in range(landmarksNos):
#             # print(pos)
#             total_pos[pos] += (numpylandmarks[pos]-numpylandmarks[8])
#             # print(numpylandmarks[pos],numpylandmarks[8],numpylandmarks[pos]-numpylandmarks[8])
#             img_pos[count - 1][pos] = (numpylandmarks[pos]-numpylandmarks[8])
#     avg_pos = []
#     print(total_pos)
#     temp_tot=[np.zeros(shape=2,dtype=int) for _ in range(len(img_pos[0]))]
#     for imgval in img_pos:
#         for posi in range(len(imgval)):
#             # print(temp_tot[posi])
#             temp_tot[posi]+=imgval[posi]
#             # print('a',temp_tot[posi],temp_tot)
#     print(temp_tot)
#     print((img_pos))
#     for pos in total_pos:
#         avg_pos.append(pos / count)
#     # print(avg_pos[0],img_pos[0][0],avg_pos[0]-img_pos[0][0])
#     max_diff = 0
#     max_diff_image = None
#     count2 = 0
#     for image in viddirectorylisting:
#         diff = []
#         # print(landmark_list,len(avg_pos))
#         for pos in landmark_list:
#             diff.append(abs(avg_pos[pos] - img_pos[count2][pos]))
#         count2 += 1
#         # print(sum(diff),len(diff),sum(sum(diff)/len(diff)))
#         avg_diff = sum(sum(diff) / len(diff))
#         # print(max_diff,avg_diff)
#         if max_diff < avg_diff:
#             max_diff = avg_diff
#             max_diff_image = image
#     return max_diff,max_diff_image


path='../../CASMEII_categorical/'


targetpath= '../../CASMEII_categorical_apex_SelectiveDivideAndConquer_NEW_mod/'
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
counter01=1

landmark_list = [x for x in range(17, 27)]
landmark_list.extend([x for x in range(36, 68)])

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
        print(counter01)
        print(videopath,viddirectorylisting)
        viddirectorylisting.sort()
        print(videopath, viddirectorylisting)
        count=0
        Lmax=0
        Rmax=0
        Lval=None
        Rval=None
        lenght=len(viddirectorylisting)
        imgs=viddirectorylisting
        img_pos=generate_pos(imgs,landmark_list)
        start=0
        end=len(imgs)
        while lenght>2:
            Lmax,Lval=new_find_max(start,start+int((end-start)//2),img_pos,imgs,landmark_list)
            Rmax,Rval = new_find_max(start+int((end-start)//2)+1,end,img_pos,imgs,landmark_list)
            if Lmax>Rmax:
                end=start+int((end-start)//2)
            else:
                start=start+int((end-start)//2)
            lenght=end-start
        if Lmax > Rmax:
            max_diff_image=Lval
            max_diff=Lmax
        else:
            max_diff_image=Rval
            # max_diff=Rmax
        # print(max_diff_image,max_diff)
        for image in viddirectorylisting:
            if viddirectorylisting[0] == image :
                # print(vidid, image,1)
                # print(videopath + "/" + str(image))
                shutil.copyfile(videopath + "/" + str(image),
                                newvideopath+ '/1' + str(image))
            if  max_diff_image == image:
                # print(vidid,image,2)
                # print(videopath+"/"+str(image))
                shutil.copyfile(videopath+"/"+str(image), newvideopath+'/2'+str(image))
        counter01+=1

        #         if found==True:
        #             print("multiple",video,catdata[vidid])
        #         found=True
        #         count+=1
        # if found==False:
        #     print("not found",video)
        # # print(str(subjectdirectorylisting)+str(video))
        # continue

