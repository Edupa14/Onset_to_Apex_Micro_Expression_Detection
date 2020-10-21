import os



path='../../../SAMM_categorical/'



directorylisting = os.listdir(path)
img_count=[]
for subject in directorylisting:
    # print(subject)
    subjectdirectorylisting=os.listdir(path+subject)
    c = 0
    for video in subjectdirectorylisting:
        videopath = path + subject + '/' + video
        # print(videopath)
        imgs = os.listdir(videopath)
        count = 0
        for img in imgs:
            count += 1
        # print(count)
        img_count.append(count)
        c += 1
    print(subject, c)
print(img_count)
print(sum(img_count)/len(img_count))
print(max(img_count))
print(min(img_count))