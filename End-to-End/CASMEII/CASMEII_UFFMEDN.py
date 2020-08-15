import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LeakyReLU ,PReLU
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from keras import backend as K
from keras.optimizers import Adam,SGD
import statistics as stat
import math
import os

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss') <= 0.1):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(1.0*100))
            self.model.stop_training = True

def new_evaluate(segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,test_index,segment_train_labels_cat, segment_validation_labels_cat ):

    model = Sequential()
    #model.add(ZeroPadding3D((2,2,0)))
    model.add(
        Convolution3D(32, (5, 5, 3), strides=(5, 5, 3), input_shape=(1, sizeH, sizeV, sizeD), padding='Same'))
    model.add(PReLU())
    # model.add(Dropout(0.5))
    model.add(
        Convolution3D(32, (3, 3, 2), strides=(3, 3, 2), padding='Same'))
    model.add(PReLU())
    model.add(
        Convolution3D(32, (3, 3, 2), strides=1, padding='Same'))
    model.add(PReLU())
    # model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add( PReLU())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, init='normal'))
    model.add(Dropout(0.5))
    # model.add(Dense(128, init='normal'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))
    # model.add(Dropout(0.5))
    # model.add(Activation('softmax'))
    opt = Adam(lr=0.001)
    model.compile(loss="mean_squared_logarithmic_error", optimizer=opt)

    model.summary()

    filepath="weights_CASMEII/weights-improvement"+str(test_index)+"-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True, verbose=1, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,cooldown=5, verbose=1,min_delta=0, mode='min',min_lr=0.0005)
    callbacks_list = [ EarlyStop, reduce,myCallback()]






    # Training the model

    history = model.fit(segment_train_images, segment_train_labels, validation_data = (segment_validation_images, segment_validation_labels), callbacks=callbacks_list, batch_size = 128, nb_epoch = 500, shuffle=True,verbose=1)








    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict([segment_validation_images])
    # predictions_labels = numpy.argmax(predictions, axis=1)
    # validation_labels = numpy.argmax(segment_validation_labels, axis=1)
    print("----------------")
    print(predictions)
    print(segment_validation_labels)
    # cfm = confusion_matrix(validation_labels, predictions_labels)
    # print (cfm)
    # print("accuracy: ",accuracy_score(validation_labels, predictions_labels))
    print("----------------")
    print(segment_train_images.shape,segment_train_labels.shape)
    #generate apex data for train
    segment_train_images_cat=numpy.zeros(segment_train_images.shape[0])
    for i in range(segment_train_images.shape[0]):
        print(segment_train_labels[i])
        # print(segment_train_images[i])
        segment_train_images_cat[i][0][:][:][:]=segment_train_images[i,:,:,:,segment_train_labels[i]]
    print("sssssssssssssssssssssssssssssssssss",segment_train_images_cat.shape)


    return segment_validation_labels,predictions





#-----------------------------------------------------------------------------------------------------------------
#LOOCV
def loocv():
    loo = LeaveOneOut()
    loo.get_n_splits(segment_training_set)
    tot=0
    count=0
    accs=[]
    accs2=[]
    vals=[]
    preds=[]
    diffs=[]
    for train_index, test_index in loo.split(segment_training_set):

        # print(segment_traininglabels[train_index])
        # print(segment_traininglabels[test_index])
        print(test_index)
        val,pred = evaluate(segment_training_set[train_index], segment_training_set[test_index],segment_traininglabels[train_index], segment_traininglabels[test_index] ,test_index)
        # tot+=val_acc
        for i in range(val.shape[0]):
            diffs.append(abs(val[i] - int(round(pred[i][0]))))
        # accs.append(val_acc)
        accs2.append(segment_traininglabels[test_index])
        count+=1
        print("------------------------------------------------------------------------")
        # print("validation acc:",val_acc)
        print("current average diffs:",(sum(diffs)) / len(diffs))
        print("------------------------------------------------------------------------")
    print(tot/count)
    print('depth: ',sizeD)
    print(accs)
    print('9')
    print(segmentName)
    print(diffs)
    print("MAE: ",(sum(diffs))/len(diffs))
    print("SD: ",stat.stdev(diffs))
    print("SE: ",stat.stdev(diffs)/(math.sqrt(len(diffs))))
    return (sum(diffs))/len(diffs) , stat.stdev(diffs)/(math.sqrt(len(diffs)))





#-----------------------------------------------------------------------------------------------------------------
#Test train split

def split():
    # Spliting the dataset into training and validation sets
    segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels = train_test_split(segment_training_set,
                                                                                                segment_traininglabels,
                                                                                                test_size=0.2,random_state=1)

    # Save validation set in a numpy array
    numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV), segment_validation_images)
    numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV), segment_validation_labels)

    # Loading Load validation set from numpy array
    #
    # eimg = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
    # labels = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))

    val,pred=evaluate(segment_train_images, segment_validation_images,segment_train_labels, segment_validation_labels ,0)
    diffs=[]
    for i in range(val.shape[0]):
        diffs.append(abs(val[i] - int(round(pred[i][0]))))
    print("MAE: ", (sum(diffs)) / len(diffs))
    print("SD: ", stat.stdev(diffs))
    print("SE: ", stat.stdev(diffs) / (math.sqrt(len(diffs))))
    return (sum(diffs)) / len(diffs), stat.stdev(diffs) / (math.sqrt(len(diffs)))


#-----------------------------------------------------------------------------------------------------------------
#k-fold
def kfold():
    kf = KFold(n_splits=2, random_state=42, shuffle=True)
    tot=0
    count=0
    accs=[]
    accs2=[]
    vals=[]
    preds=[]
    diffs = []
    for train_index, test_index in kf.split(segment_training_set):

        # print(segment_traininglabels[train_index])
        # print(segment_traininglabels[test_index])
        print(test_index)
        val,pred = new_evaluate(segment_training_set[train_index], segment_training_set[test_index],segment_traininglabels[train_index], segment_traininglabels[test_index] ,test_index,tempsegment_traininglabels_cat,tempsegment_traininglabels_cat)
        # tot+=val_acc
        for i in range(val.shape[0]):
            diffs.append(abs(val[i] - int(round(pred[i][0]))))
        # accs.append(val_acc)
        accs2.append(segment_traininglabels[test_index])
        count+=1


    print("MAE: ", (sum(diffs)) / len(diffs))
    print("SD: ", stat.stdev(diffs))
    print("SE: ", stat.stdev(diffs) / (math.sqrt(len(diffs))))
    print(diffs)
    print("MAE: ",(sum(diffs))/len(diffs))
    print("SD: ",stat.stdev(diffs))
    print("SE: ",stat.stdev(diffs)/(math.sqrt(len(diffs))))
    return (sum(diffs)) / len(diffs), stat.stdev(diffs) / (math.sqrt(len(diffs)))


####################################
# edit params
K.set_image_dim_ordering('th')

segmentName = 'UpperFaceDual'
sizeH=32
sizeV=32
sizeD=140

testtype = "kfold"
####################################

# Load training images and labels that are stored in numpy array

segment_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
tempsegment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
tempsegment_traininglabels_cat = []
    # numpy.load('numpy_training_datasets/{0}_labels_cat_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))

segment_traininglabels=[]
for item in tempsegment_traininglabels:
    for cat in range(len(item)):
        if item[cat]==1:
            segment_traininglabels.append(cat)
# uniques=[x for x in range(sizeD)]
# print(len(tempsegment_traininglabels),len(tempsegment_traininglabels[0]))
# segment_training_data =[numpy.argmax(y, axis=None, out=None) for y in tempsegment_traininglabels[:]]

print(segment_traininglabels,len(segment_traininglabels),len(segment_training_set))
segment_traininglabels= numpy.array(segment_traininglabels)
print(segment_traininglabels,len(segment_traininglabels),len(segment_training_set))

# print(segment_traininglabels)
# print(numpy.sum(segment_traininglabels,axis=0))



if testtype == "kfold":
    mae,se = kfold()
elif testtype == "loocv":
    mae,se = loocv()
elif testtype == "split":
    mae,se = split()
else:
    print("error")

# ---------------------------------------------------------------------------------------------------
# write to results

results = open("../TempResults.txt", 'a')
results.write("---------------------------\n")
full_path = os.path.realpath(__file__)
results.write(
    str(os.path.dirname(full_path)) + " {0}_{1}_{2}x{3}x{4}\n".format(testtype, segmentName, sizeH, sizeV, sizeD))
results.write("---------------------------\n")
results.write("MAE: " + str(mae)+ "\n")
results.write("SE:" + str(se) + "\n")