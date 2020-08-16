import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LeakyReLU ,PReLU,concatenate,Input
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from keras import backend as K
from keras.optimizers import Adam,SGD
import statistics as stat
import math
import os

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss') <= 0.0001):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(1.0*100))
            self.model.stop_training = True
class myCallback2(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc') >= 1.0):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(1.0*100))
            self.model.stop_training = True

def new_evaluate(segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,test_index,segment_train_labels_cat, segment_validation_labels_cat ):

    # generate apex data for train
    segment_train_images_cat = numpy.zeros((segment_train_images.shape[0], segment_train_images.shape[1],
                                            segment_train_images.shape[2], segment_train_images.shape[3]))
    # print(segment_train_images_cat)
    for i in range(segment_train_images.shape[0]):
        # print(segment_train_labels[i])
        # print(segment_train_images[i])
        # print(segment_train_images[i,:,:,:,[segment_train_labels[i]]].shape)
        segment_train_images_cat[i][:][:][:] = segment_train_images[i, :, :, :, segment_train_labels[i]]

    segment_train_images_cat2 = numpy.zeros((segment_train_images.shape[0], segment_train_images.shape[1],
                                             segment_train_images.shape[2], segment_train_images.shape[3]))
    # print(segment_train_images_cat2)
    print("aaaaaaaaaaaaaaaaaas", segment_train_images_cat.shape)
    for i in range(segment_train_images.shape[0]):
        # print(segment_train_labels[i])
        # print(segment_train_images[i])
        # print(segment_train_images[i, :, :, :, [segment_train_labels[i]]].shape)
        segment_train_images_cat2[i][:][:][:] = segment_train_images[i, :, :, :, 0]
    segment_train_images_cat = numpy.stack([ segment_train_images_cat2,segment_train_images_cat], axis=4)
    # print("sssssssssssssssssssssssssssssssssss", segment_train_images_cat.shape)
    #
    # print(segment_train_images[0, :, :, :, 0])
    # print(segment_train_images_cat[0, :, :, :, 0])



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







    # generate apex data for test
    segment_test_images_cat = numpy.zeros((segment_validation_images.shape[0], segment_validation_images.shape[1],
                                            segment_validation_images.shape[2], segment_validation_images.shape[3]))
    # print(segment_train_images_cat)
    for i in range(segment_validation_images.shape[0]):
        # print(segment_train_labels[i])
        # print(segment_train_images[i])
        # print(segment_train_images[i,:,:,:,[segment_train_labels[i]]].shape)
        segment_test_images_cat[i][:][:][:] = segment_validation_images[i, :, :, :,int(float(predictions[i]))]

    segment_test_images_cat2 = numpy.zeros((segment_validation_images.shape[0], segment_validation_images.shape[1],
                                             segment_validation_images.shape[2], segment_validation_images.shape[3]))
    # print(segment_train_images_cat2)
    print("aaaaaaaaaaaaaaaaaas", segment_train_images_cat.shape)
    for i in range(segment_validation_images.shape[0]):
        # print(segment_train_labels[i])
        # print(segment_train_images[i])
        # print(segment_train_images[i, :, :, :, [segment_train_labels[i]]].shape)
        segment_test_images_cat2[i][:][:][:] = segment_validation_images[i, :, :, :, 0]
    segment_validation_images_cat = numpy.stack([segment_test_images_cat2, segment_test_images_cat], axis=4)
    print("sssssssssssssssssssssssssssssssssss", segment_test_images_cat.shape)
    #
    # print(segment_train_images[0, :, :, :, 0])
    # print(segment_train_images_cat[0, :, :, :, 0])

    #-------------------------------------------
    layer_in = Input(shape=(1, sizeH, sizeV, 2))
    # conv1 = Convolution3D(256, (20, 20, 9), strides=(10, 10, 3), padding='Same')(input)
    # # bn1=BatchNormalization()(conv1)
    # ract_1 = PReLU()(conv1)
    conv1 = Convolution3D(96, (20, 20, 1), strides=(10, 10, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Convolution3D(256, (20, 20, 1), strides=(10, 10, 1), padding='same', activation='relu')(layer_in)
    conv3 = Convolution3D(512, (3, 3, 1), padding='same', activation='relu')(conv3)
    # 5x5 conv
    # conv5 = Convolution3D(96, (20, 20, 1), strides=(10, 10, 1), padding='same', activation='relu')(layer_in)
    # conv5 = Convolution3D(128, (5, 5, 1), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(layer_in)
    pool = Convolution3D(32, (20, 20, 1), strides=(10, 10, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3,  pool], axis=-4)
    # add1= Add() ([conv3,ract_1])
    drop0 = Dropout(0.5)(layer_out)
    # conv6 = Convolution3D(512, (3, 3, 3), strides=1, padding='Same')(drop0)
    # # bn3 = BatchNormalization()(conv3)
    ract_4 = PReLU()(drop0)
    flatten_1 = Flatten()(ract_4)
    # dense_1 = Dense(1024, init='normal')(flatten_1)
    # dense_2 = Dense(128, init='normal')(dense_1)



    layer_in2 = Input(shape=(1, sizeH, sizeV, sizeD))
    conv21 =  Convolution3D(32, (20, 20,50), strides=(10, 10, 25), padding='Same')(layer_in2)
    ract_21 = PReLU()(conv21)
    conv22 = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_21)
    ract_22 = PReLU()(conv22)
    flatten_2 = Flatten()(ract_22)

    drop11 = Dropout(0.5)(flatten_1)
    drop21 = Dropout(0.5)(flatten_2)
    concat = concatenate([drop11,drop21,layer_in2], axis=-1)

    dense_3 = Dense(5, init='normal')(concat)
    # drop1 = Dropout(0.5)(dense_3)
    activation = Activation('softmax')(dense_3)
    opt = SGD(lr=0.01)
    model_cat = Model(inputs=[layer_in,layer_in2], outputs=activation)
    model_cat.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#__________________________________________
    # model_cat = Sequential()
    # # model_cat.add(ZeroPadding3D((2,2,0)))
    # model_cat.add(Convolution3D(32, (9, 9, 1), strides=(3, 3, 1), input_shape=(1, sizeH, sizeV, 2), padding='Same'))
    #
    # # model_cat.add(Convolution3D(64, (12, 12, 1), strides=1, input_shape=(1, sizeH, sizeV, sizeD), padding='Same'))
    # model_cat.add(PReLU())
    # # model_cat.add(Convolution3D(128, (8, 8, 1), strides=1, input_shape=(1, sizeH, sizeV, sizeD), padding='Same'))
    # # model_cat.add(PReLU())
    # # model_cat.add(Dropout(0.5))
    # # 3
    # # model_cat.add(Convolution3D(32, (3, 3, 2), strides=1, padding='Same'))
    # # model_cat.add(PReLU())
    # # 40
    # # model_cat.add(Dropout(0.5))
    # # 1
    # model_cat.add(MaxPooling3D(pool_size=(3, 3, 2)))
    # model_cat.add(PReLU())
    # # 2
    # # model_cat.add(Dropout(0.5))
    # model_cat.add(Flatten())
    # model_cat.add(Dense(256, init='normal'))
    # # model_cat.add(Dropout(0.5))
    # model_cat.add(Dense(128, init='normal'))
    # # model_cat.add(PReLU())
    # # model_cat.add(Dense(128, init='normal'))`
    # model_cat.add(Dropout(0.5))
    # model_cat.add(Dense(5, init='normal'))
    # # model.add(Dropout(0.5))
    # model_cat.add(Activation('softmax'))
    # opt = SGD(lr=0.1)
    # model_cat.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model_cat.summary()

    filepath = "weights_CAS(ME)2/weights-improvement" + str(test_index) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    EarlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, restore_best_weights=True, verbose=1,
                              mode='max')
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, cooldown=5, verbose=1, min_delta=0,
                               mode='max', min_lr=0.0005)
    callbacks_list_cat = [EarlyStop, reduce, myCallback2()]

    # Training the model

    history2= model_cat.fit([segment_train_images_cat,segment_train_images], segment_train_labels_cat,
                        validation_data=([segment_validation_images_cat,segment_validation_images], segment_validation_labels_cat),
                        callbacks=callbacks_list_cat, batch_size=8, nb_epoch=500, shuffle=True, verbose=1)

    # Finding Confusion Matrix using pretrained weights

    predictions = model_cat.predict([segment_validation_images_cat,segment_validation_images])
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(segment_validation_labels_cat, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print(cfm)
    print("accuracy: ", accuracy_score(validation_labels, predictions_labels))
    return accuracy_score(validation_labels, predictions_labels), validation_labels, predictions_labels





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
    segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,segment_traininglabes_cat,segment_validlabels_cat = train_test_split(segment_training_set,
                                                                                                segment_traininglabels,tempsegment_traininglabels_cat,
                                                                                                test_size=0.1,random_state=42)

    # Save validation set in a numpy array
    # numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV), segment_validation_images)
    # numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV), segment_validation_labels)

    # Loading Load validation set from numpy array
    #
    # eimg = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
    # labels = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))

    _, val_labels, pred_labels =new_evaluate(segment_train_images, segment_validation_images,segment_train_labels, segment_validation_labels ,0,segment_traininglabes_cat,segment_validlabels_cat)
    return val_labels, pred_labels


#-----------------------------------------------------------------------------------------------------------------
#k-fold
def kfold():
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    tot = 0
    count = 0
    accs = []
    accs2 = []

    val_labels = []
    pred_labels = []
    for train_index, test_index in kf.split(segment_training_set):

        # print(segment_traininglabels[train_index])
        # print(segment_traininglabels[test_index])
        print(test_index)
        val_acc, val_label, pred_label = new_evaluate(segment_training_set[train_index], segment_training_set[test_index],segment_traininglabels[train_index], segment_traininglabels[test_index] ,test_index,tempsegment_traininglabels_cat[train_index],tempsegment_traininglabels_cat[test_index])
        tot += val_acc
        val_labels.extend(val_label)
        pred_labels.extend(pred_label)
        accs.append(val_acc)
        accs2.append(segment_traininglabels[test_index])
        count += 1
        print("------------------------------------------------------------------------")
        print("validation acc:", val_acc)
        print("------------------------------------------------------------------------")
    print("accuracy: ", accuracy_score(val_labels, pred_labels))
    cfm = confusion_matrix(val_labels, pred_labels)
    # tp_and_fn = sum(cfm.sum(1))
    # tp_and_fp = sum(cfm.sum(0))
    # tp = sum(cfm.diagonal())
    print("cfm: \n", cfm)
    # print("tp_and_fn: ",tp_and_fn)
    # print("tp_and_fp: ",tp_and_fp)
    # print("tp: ",tp)
    #
    # precision = tp / tp_and_fp
    # recall = tp / tp_and_fn
    # print("precision: ",precision)
    # print("recall: ",recall)
    # print("F1-score: ",f1_score(val_labels,pred_labels,average="macro"))
    print("F1-score: ", f1_score(val_labels, pred_labels, average="weighted"))
    return val_labels, pred_labels


####################################
# edit params
K.set_image_dim_ordering('th')

segmentName = 'UpperFace'
sizeH=128
sizeV=128
sizeD=140

testtype = "split"
####################################

# Load training images and labels that are stored in numpy array

segment_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
tempsegment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
tempsegment_traininglabels_cat = numpy.load('numpy_training_datasets/{0}_labels_cat_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))

print(tempsegment_traininglabels,len(tempsegment_traininglabels[0]))
print(tempsegment_traininglabels_cat,tempsegment_traininglabels_cat[0].size)
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
    val_labels, pred_labels = kfold()
elif testtype == "loocv":
    val_labels, pred_labels = loocv()
elif testtype == "split":
    val_labels, pred_labels= split()
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
results.write("accuracy: " + str(accuracy_score(val_labels, pred_labels)) + "\n")
results.write("F1-score: " + str(f1_score(val_labels, pred_labels, average="weighted")) + "\n")