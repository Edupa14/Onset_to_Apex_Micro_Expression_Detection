
catdatafile = pd.read_excel('../../cat_apex.xlsx')
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

segment_traininglabels = np_utils.to_categorical(segment_traininglabels, sizeD)