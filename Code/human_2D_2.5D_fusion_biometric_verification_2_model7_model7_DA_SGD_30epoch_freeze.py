# verification_3_DA_stratify
# 4 i/p to network
from numpy import mean, std
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
#from keras.utils import to_categorical
#import keras
import numpy as np
import random
import os.path
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, auc, roc_curve, RocCurveDisplay
import sys
import cv2
from glob import glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='0'


IMG_SHAPE = (224, 224, 1)
# train_dir = '../../trial data/Training'
all_images_cropped_2D = '../../UND 2D cropped face data/cropped_data/*.ppm'
dataAugmentation_crop_2D = '../../UND 2D cropped face data/cropped_data_DA_ver3/*_cropped.ppm'
dataAugmentation_rotate_2D = '../../UND 2D cropped face data/cropped_data_DA_ver3/*_rotated*.ppm'
noise_augmentation_2D = '../../UND 2D cropped face data/cropped_data_DA_ver3/*_noise.ppm'


all_images_cropped_25D = '../../UND 2.5D cropped face data/2.5D Cropped Data/*.jpg'
dataAugmentation_crop_25D = '../../UND 2.5D cropped face data/2.5D_cropped_data_DA/*_cropped.jpg'
dataAugmentation_rotate_25D = '../../UND 2.5D cropped face data/2.5D_cropped_data_DA/*_rotated*.jpg'
noise_augmentation_25D = '../../UND 2.5D cropped face data/2.5D_cropped_data_DA/*_noise.jpg'
#categories_n = 200
# classes = [x for x in range(200)]

images_2D = []
images_25D = []
targets_2D = []
targets_25D = []
labels_2D = []
labels_25D = []
img_names_2D = []
img_names_25D=[]


def load_dataset(path, images, targets, labels, img_names_list):
    img_names = sorted(glob(path))
    for fn in img_names:
        img = cv2.imread(fn)
        img = cv2.resize(img, (224, 224))
        # img = load_img(fn)
        # print(img.shape)
        images.append(img)
        name=fn.split("/")[-1]
        label = fn.split("/")[-1].split("d")[0]  # \\ for windows, / for linux
        img_names_list.append(fn.split("/")[-1])
        if label not in labels:
            labels.append(label)
        target = labels.index(label)
        targets.append([target,name])
    return images, targets, labels


images_2D, targets_2D, labels_2D = load_dataset(all_images_cropped_2D, images_2D, targets_2D, labels_2D,img_names_2D)
images_2D, targets_2D, labels_2D = load_dataset(dataAugmentation_crop_2D, images_2D, targets_2D, labels_2D,img_names_2D)
images_2D, targets_2D, labels_2D = load_dataset(dataAugmentation_rotate_2D, images_2D, targets_2D, labels_2D,img_names_2D)

images_25D, targets_25D, labels_25D = load_dataset(all_images_cropped_25D, images_25D, targets_25D, labels_25D, img_names_25D)
images_25D, targets_25D, labels_25D = load_dataset(dataAugmentation_crop_25D, images_25D, targets_25D, labels_25D, img_names_25D)
images_25D, targets_25D, labels_25D = load_dataset(dataAugmentation_rotate_25D, images_25D, targets_25D, labels_25D, img_names_25D)

"""print("img names 2D == img names 2.5D: ", img_names_2D == img_names_25D)
print("img_names_2D: ",img_names_2D[:12])
print("img_names_2.5D: ",img_names_25D[:12])"""

noise_images_2D = []
noise_targets_2D = []
noise_img_names_2D = []
noise_images_2D, noise_targets_2D, labels_2D = load_dataset(
    noise_augmentation_2D, noise_images_2D, noise_targets_2D, labels_2D, noise_img_names_2D)

noise_images_25D = []
noise_targets_25D = []
noise_img_names_25D = []
noise_images_25D, noise_targets_25D, labels_25D = load_dataset(
    noise_augmentation_25D, noise_images_25D, noise_targets_25D, labels_25D, noise_img_names_25D)

images_num = len(images_2D)
categories_n=len(labels_2D)
"""targets = [int(ele) - 1 for ele in targets]
noise_targets = [int(ele) - 1 for ele in noise_targets]"""
print("Number of total samples = ", images_num)
print("Number of Noisy images = ", len(noise_images_2D))
print("Number of subjects = ", categories_n)

# print(targets)

# convert to grayscale


def convert_gray(images):
    gray_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray_img = cv2.equalizeHist(gray_img)
        gray_images.append(gray_img)
    return gray_images


gray_images_2D = convert_gray(images_2D)
noise_gray_images_2D = convert_gray(noise_images_2D)
gray_images_25D = convert_gray(images_25D)
noise_gray_images_25D = convert_gray(noise_images_25D)

# cv2.imwrite("../../stretched.png", gray_images[2])
# split into training, testing and validation
mystate=41
x_train_2D, x_test_2D, y_train_2D, y_test_2D = train_test_split(
    gray_images_2D, targets_2D, test_size=0.3, shuffle=True, random_state=mystate)
x_val_2D, x_test_2D, y_val_2D, y_test_2D = train_test_split(
    x_test_2D, y_test_2D, test_size=0.5, shuffle=True, random_state=mystate)

x_train_25D, x_test_25D, y_train_25D, y_test_25D = train_test_split(
    gray_images_25D, targets_25D, test_size=0.3, shuffle=True, random_state=mystate)
x_val_25D, x_test_25D, y_val_25D, y_test_25D = train_test_split(
    x_test_25D, y_test_25D, test_size=0.5, shuffle=True, random_state=mystate)

"""print("Testing targets 2D == Testing targets 2.5D: ", y_test_2D == y_test_25D)
print("y_test_2D: ",y_test_2D[:12])
print("y_test_2.5D: ",y_test_25D[:12])"""

x_train_2D.extend(noise_gray_images_2D)
y_train_2D.extend(noise_targets_2D)
x_train_25D.extend(noise_gray_images_25D)
y_train_25D.extend(noise_targets_25D)
temp = list(zip(x_train_2D, y_train_2D,x_train_25D,y_train_25D))
random.shuffle(temp)
x_train_2D, y_train_2D,x_train_25D,y_train_25D = zip(*temp)

#y_train_ohe = to_categorical(y_train, categories_n)
#y_val_ohe = to_categorical(y_val, categories_n)


# preprocessing

# x_train = np.array(convert_img_to_array(x_train))
x_train_2D = np.array(x_train_2D)
print('2D Training set shape : ', x_train_2D.shape)
# x_val = np.array(convert_img_to_array(x_val))
x_val_2D = np.array(x_val_2D)
print('2D Validation set shape : ', x_val_2D.shape)
# x_test = np.array(convert_img_to_array(x_test))
x_test_2D = np.array(x_test_2D)
print('2D Test set shape : ', x_test_2D.shape)

x_train_25D = np.array(x_train_25D)
print('2.5D Training set shape : ', x_train_25D.shape)
# x_val = np.array(convert_img_to_array(x_val))
x_val_25D = np.array(x_val_25D)
print('2.5D Validation set shape : ', x_val_25D.shape)
# x_test = np.array(convert_img_to_array(x_test))
x_test_25D = np.array(x_test_25D)
print('2.5D Test set shape : ', x_test_25D.shape)
"""print("Train targets 2D == Train targets 2.5D: ", y_train_2D == y_train_25D)
print("Validation targets 2D == Validation targets 2.5D: ", y_val_2D == y_val_25D)
print("Testing targets 2D == Testing targets 2.5D: ", y_test_2D == y_test_25D)
print("y_test_2D: ",y_test_2D)
print("y_test_2.5D: ",y_test_25D)"""

x_train_2D = x_train_2D.reshape((x_train_2D.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))
x_val_2D = x_val_2D.reshape((x_val_2D.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))
x_test_2D = x_test_2D.reshape((x_test_2D.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))

x_train_25D = x_train_25D.reshape((x_train_25D.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))
x_val_25D = x_val_25D.reshape((x_val_2D.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))
x_test_25D = x_test_25D.reshape((x_test_25D.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))

for img in x_train_2D:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_val_2D:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_test_2D:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_train_25D:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_val_25D:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_test_25D:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

# cv2.imwrite("../../stretched2.jpg", x_train[2])
# normalization
x_train_2D = x_train_2D.astype('float32')
x_train_2D = x_train_2D/255
x_val_2D = x_val_2D.astype('float32')
x_val_2D = x_val_2D/255
x_test_2D = x_test_2D.astype('float32')
x_test_2D = x_test_2D/255

x_train_25D = x_train_25D.astype('float32')
x_train_25D = x_train_25D/255
x_val_25D = x_val_25D.astype('float32')
x_val_25D = x_val_25D/255
x_test_25D = x_test_25D.astype('float32')
x_test_25D = x_test_25D/255

def make_pairs(images_2D,images_25D, targets):
    pairImages = []
    pairLabels = []
    pairNames=[]
    targets=np.array(targets)
    labels=np.array(targets[:,0], dtype = np.int32)
    names=np.array(targets[:,1])
    # print(labels)
    numClasses = len(np.unique(labels))
    print("Number of classes: ",numClasses)
    # print(len(labels))
    labels = np.array(labels)
    idx = []
    same_count=0

    for i in range(np.max(labels)+1):
        idx.append(list(np.where(labels == i)))
    # print(idx[0][0])
    #print(len(idx))
    for idxA in range(len(images_2D)):
        currentImage_2D = images_2D[idxA]
        currentImage_25D = images_25D[idxA]
        label = labels[idxA]
        #print(label)
        #print(idx[label][0])
        # randomly pick an image that belongs to the *same* class
        # label
        label_idxs=idx[label][0]
        if(len(label_idxs)>1):
            label_idxs=np.delete(label_idxs,np.where(label_idxs==idxA))
        posIdx = np.random.choice(label_idxs)
        posImage_2D = images_2D[posIdx]
        posImage_25D = images_25D[posIdx]
        #print(idxA,posIdx)
        if (idxA==posIdx):
            print("same image: ",globals()['labels_2D'][label])
            print("Image name: ",names[idxA])
            same_count+=1
        if(posIdx==None):
            print("None")

        # prepare a positive pair and update the images and labels
        pairImages.append([currentImage_2D, posImage_2D,currentImage_25D,posImage_25D])
        pairLabels.append([1])
        pairNames.append([names[idxA],names[posIdx]])
        

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label

        negIdx = np.where(labels != label)[0]
        negImage_2D = images_2D[np.random.choice(negIdx)]
        negImage_25D = images_25D[np.random.choice(negIdx)]

        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage_2D, negImage_2D,currentImage_25D,negImage_25D])
        pairLabels.append([0])
        pairNames.append([names[idxA],names[negIdx]])
    pairImages = np.array(pairImages)
    #print(pairImages.shape)
    #pairImages=pairImages.reshape((2, 480, 640, 1))
    pairLabels = np.array(pairLabels)
    pairNames=np.array(pairNames)
    #print(pairLabels.shape)
    #pairLabels=pairLabels.reshape((2, 480, 640, 1))
    print("number of pairs having same image: ",same_count)
    return (pairImages, pairLabels,pairNames)

print("Training pairs:")
pairTrain, labelTrain, nameTrain = make_pairs(x_train_2D, x_train_25D, y_train_2D)
print("Validation pairs:")
pairVal, labelVal, nameVal = make_pairs(x_val_2D, x_val_25D, y_val_2D)
print("Testing pairs:")
pairTest, labelTest, nameTest = make_pairs(x_test_2D, x_test_25D, y_test_2D)


"""def model_config(input_shape):
    input_2D_left = Input(input_shape)
    input_2D_right = Input(input_shape)
    input_25D_left = Input(input_shape)
    input_25D_right = Input(input_shape)

    model_2D=load_model('../Models/human_2D_verification_model7_DA_2.h5')
    model_2D._name="model_2D"
    model_25D=load_model('../Models/human_2.5D_verification_model2_DA_transfer.h5')
    model_25D._name="model_2.5D"

    for layer in model_25D.layers:
        layer._name = layer.name + str("_2")
    
    encoded_2D = model_2D([input_2D_left, input_2D_right])
    encoded_25D = model_25D([input_25D_left,input_25D_right])

    merged = keras.layers.Concatenate(axis=1)([encoded_2D,1-encoded_2D,encoded_25D,1-encoded_25D])
    #print(merged.shape)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(merged)

    # Connect the inputs with the outputs
    fusion_net = Model(inputs=[input_2D_left,input_2D_right, input_25D_left,input_25D_right], outputs=prediction, name='fusion_2D-2.5D_model2')
    # model.add(Dense(200, activation='softmax'))
    # compile the model
    opt = keras.optimizers.Adam(lr=0.001)
    fusion_net.compile(optimizer=opt, loss='binary_crossentropy',
                        metrics=['accuracy'])
    return fusion_net"""


def train_model(datax, datay, valx, valy):
    model_2D=load_model('../Models/human_2D_verification_model7_DA_2.h5')
    model_2D._name="model_2D"
    model_2D.optimizer._name=model_2D.optimizer._name+str("_2D")
    model_25D=load_model('../Models/human_2.5D_verification_model7_DA_2.h5')
    model_25D._name="model_2.5D"
    model_25D.optimizer._name=model_25D.optimizer._name+str("_25D")

    model_2D.trainable=False
    model_25D.trainable=False

    for layer in model_25D.layers:
        layer._name = layer.name + str("_2")
    
    model_2D.layers[0]._name="input_2D_left"
    model_2D.layers[1]._name="input_2D_right"
    model_25D.layers[0]._name="input_25D_left"
    model_25D.layers[1]._name="input_25D_right"

    encoded_2D = model_2D.layers[-1].output
    encoded_25D = model_25D.layers[-1].output

    merged = tensorflow.keras.layers.Concatenate(axis=1)([encoded_2D,1-encoded_2D,encoded_25D,1-encoded_25D])
    #print(merged.shape)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(merged)

    # Connect the inputs with the outputs
    model = Model(inputs=model_2D.input + model_25D.input, outputs=prediction, name='fusion_2D-2.5D_model2')
    # model.add(Dense(200, activation='softmax'))
    # compile the model
    #opt = keras.optimizers.Adam(lr=0.001,name="Adam_fusion")
    opt = SGD(lr=0.01, momentum=0.9, name="SGD_fusion")
    model.compile(optimizer=opt, loss='binary_crossentropy',
                        metrics=['accuracy'])
    model.summary()
    #keras.utils.plot_model(model,"../../Outputs/fusion_2D-2.5D_model_2_model2_DA_transfer.png",show_shapes=True)
    # fit model
    history = model.fit([datax[:, 0], datax[:, 1],datax[:,2], datax[:,3]], datay[:], epochs=30, batch_size=32,
                        validation_data=([valx[:, 0], valx[:, 1],valx[:,2],valx[:,3]], valy), verbose=2)
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['accuracy'])*100, std(history.history['accuracy'])*100, len(history.history['accuracy'])))
    # print('Top-5 Accuracy: mean=%.3f std=%.3f, n=%d' %
    #      (mean(history.history['top_k_categorical_accuracy'])*100, std(history.history['top_k_categorical_accuracy'])*100, len(history.history['top_k_categorical_accuracy'])))
    print('Validation Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_accuracy'])*100, std(history.history['val_accuracy'])*100, len(history.history['val_accuracy'])))
    # print('Validation Top-5 Accuracy: mean=%.3f std=%.3f, n=%d' %
    #      (mean(history.history['val_top_k_categorical_accuracy'])*100, std(history.history['val_top_k_categorical_accuracy'])*100, len(history.history['val_top_k_categorical_accuracy'])))

    return history, model


history, model = train_model(pairTrain, labelTrain, pairVal, labelVal)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Binary Crossentropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig('../../Outputs/verification_2D_2.5D_fusion_model7_model7_DA_SGD_30epoch_freeze_graph.png')
plt.close()


# evaluation
print("Model evaluation on test dataset: ")
pred_prob = model.predict([pairTest[:, 0], pairTest[:, 1],pairTest[:,2],pairTest[:,3]])
# pred_class = model.predict_classes(x_test)
pred_class = (pred_prob > 0.5).astype("int32")

# metrics
accuracy = accuracy_score(labelTest, pred_class)
# k_accuracy = top_k_accuracy_score(y_test, pred_prob, k=5)
print('accuracy =  %.3f' % (accuracy * 100.0))
# 'top-5 accuracy = %.3f' % (k_accuracy*100))
# precision = precision_score(y_test, pred_class)
# recall = recall_score(y_test, pred_class)

report = classification_report(labelTest, pred_class)
print("Classification Report: ")
print(report)
"""f1 = f1_score(y_test, pred_class,average='macro')
print("f1 score: ", f1)"""

# calculate AUC
auc = roc_auc_score(labelTest, pred_prob)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(labelTest, pred_prob)


# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title('ROC curve on UND 2D-2.5D fusion')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(['No skill', 'ROC Curve'], loc='lower right')
plt.savefig('../../Outputs/verification_2D_2.5D_fusion_model7_model7_DA_SGD_30epoch_freeze_roc.png')
plt.close()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
roc_display.plot()
plt.savefig('../../Outputs/verification_2D_2.5D_fusion_model7_model7_DA_SGD_30epoch_freeze_display_roc.png')
plt.close()
fnr = 1 - tpr

eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
EER_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
EER = (fpr[np.nanargmin(np.absolute((fnr - fpr)))] +
       fnr[np.nanargmin(np.absolute((fnr - fpr)))])/2
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
print("EER = ", EER*100, "%, EER Threshold = ", eer_threshold)
print("EER_fpr = ", EER_fpr*100,"%")
"""y_test = np.array(y_test)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
for i in range(categories_n):
    y_test_bin = (y_test == i).astype(np.int32)
    y_score = pred_prob[:, i]
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)
    roc_display1 = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_display1.plot(ax=ax1)
    roc_display2 = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display2.plot(ax=ax2)
# plt.legend(['acc', 'val_acc'], loc='lower right')
# plt.savefig(os.path.join(result_dir, 'roc.png'))
fig1.savefig('../../Outputs/model19_2_roc.png')
fig2.savefig('../../Outputs/model19_2_roc_withoutAuc.png')
# show the plot
# plt.show()
plt.close(fig1)
plt.close(fig2)"""

confusionMatrix = confusion_matrix(
    labelTest, pred_class)  # row(true), column(predicted)
# np.set_printoptions(threshold=sys.maxsize)
print("Confusion matrix: ")
print(confusionMatrix)
# np.set_printoptions(threshold=False)
# cm_labels = [x for x in range(20)]
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusionMatrix)
disp.plot()

plt.savefig('../../Outputs/verification_2D_2.5D_fusion_model7_model7_DA_SGD_30epoch_freeze_confusionMatrix.png')
plt.close()
print("Flase Positive Rate = ", fpr)
print("True Positive Rate = ", tpr)
print("Thresholds = ", thresholds)
print("Flase Negative Rate = ", fnr)

fig, ax = plt.subplots()

ax.plot(thresholds, fpr, 'r--', label='FAR')
ax.plot(thresholds, fnr, 'g--', label='FRR')
plt.xlabel('Threshold')
plt.plot(eer_threshold, EER, 'ro', label='EER')
legend = ax.legend()
plt.savefig('../../Outputs/verification_2D_2.5D_fusion_model7_model7_DA_SGD_30epoch_freeze_EER.png')
plt.close()

pred_prob=pred_prob*100
same=[pred_class==labelTest]
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
np.savetxt('../../Outputs/verification_2D_2.5D_fusion_model7_model7_DA_30epoch_freeze_scores.csv', np.array(list(zip(nameTest,pred_prob,pred_class,labelTest,same))), delimiter=',', header="Image1,Image2,pred probability,pred class, actual class, same(0)")

model.save('../Models/human_2D_2.5D_fusion_verification_model7_model7_DA_30epoch_freeze.h5')
print("saved")