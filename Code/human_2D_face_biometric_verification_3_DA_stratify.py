# siamese network - biometric verification - FEI model9
# added condition in making pairs such that same image is not paired if possible
from numpy import mean, std
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input, Lambda
from keras.models import Sequential, Model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
#from keras.utils import to_categorical
import keras
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
all_images_cropped = '../../UND 2D cropped face data/cropped_data/*.ppm'
dataAugmentation_crop = '../../UND 2D cropped face data/cropped_data_DA_ver3/*_cropped.ppm'
dataAugmentation_rotate = '../../UND 2D cropped face data/cropped_data_DA_ver3/*_rotated*.ppm'
noise_augmentation = '../../UND 2D cropped face data/cropped_data_DA_ver3/*_noise.ppm'
#categories_n = 200
# classes = [x for x in range(200)]

images = []
targets = []
labels = []


def load_dataset(path, images, targets, labels):
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn)
        img = cv2.resize(img, (224, 224))
        # img = load_img(fn)
        # print(img.shape)
        images.append(img)
        label = fn.split("/")[-1].split("d")[0]  # \\ for windows, / for linux
        if label not in labels:
            labels.append(label)
        target = labels.index(label)
        targets.append(target)
    return images, targets, labels


images, targets, labels = load_dataset(all_images_cropped, images, targets, labels)
images, targets, labels = load_dataset(dataAugmentation_crop, images, targets, labels)
images, targets, labels = load_dataset(dataAugmentation_rotate, images, targets, labels)
noise_images = []
noise_targets = []
noise_images, noise_targets, labels = load_dataset(
    noise_augmentation, noise_images, noise_targets, labels)
images_num = len(images)
categories_n=len(labels)
"""targets = [int(ele) - 1 for ele in targets]
noise_targets = [int(ele) - 1 for ele in noise_targets]"""
print("Number of total samples = ", images_num)
print("Number of Noisy images = ", len(noise_images))
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


gray_images = convert_gray(images)
noise_gray_images = convert_gray(noise_images)

# cv2.imwrite("../../stretched.png", gray_images[2])
# split into training, testing and validation
mystate=41
x_train, x_test, y_train, y_test = train_test_split(
    gray_images, targets, test_size=0.3, shuffle=True, stratify=targets, random_state=mystate)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, shuffle=True, stratify=y_test, random_state=mystate)

x_train.extend(noise_gray_images)
y_train.extend(noise_targets)
temp = list(zip(x_train, y_train))
random.shuffle(temp)
x_train, y_train = zip(*temp)

#y_train_ohe = to_categorical(y_train, categories_n)
#y_val_ohe = to_categorical(y_val, categories_n)


# preprocessing

# x_train = np.array(convert_img_to_array(x_train))
x_train = np.array(x_train)
print('Training set shape : ', x_train.shape)
# x_val = np.array(convert_img_to_array(x_val))
x_val = np.array(x_val)
print('Validation set shape : ', x_val.shape)
# x_test = np.array(convert_img_to_array(x_test))
x_test = np.array(x_test)
print('Test set shape : ', x_test.shape)

x_train = x_train.reshape((x_train.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))
x_val = x_val.reshape((x_val.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))
x_test = x_test.reshape((x_test.shape[0], IMG_SHAPE[0],IMG_SHAPE[1],IMG_SHAPE[2]))

for img in x_train:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_val:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_test:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

# cv2.imwrite("../../stretched2.jpg", x_train[2])
# normalization
x_train = x_train.astype('float32')
x_train = x_train/255
x_val = x_val.astype('float32')
x_val = x_val/255
x_test = x_test.astype('float32')
x_test = x_test/255


def make_pairs(images, labels):
    pairImages = []
    pairLabels = []
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
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        #print(label)
        #print(idx[label][0])
        # randomly pick an image that belongs to the *same* class
        # label
        label_idxs=idx[label][0]
        if(len(label_idxs)>1):
            label_idxs=np.delete(label_idxs,np.where(label_idxs==idxA))
        posIdx = np.random.choice(label_idxs)
        posImage = images[posIdx]
        #print(idxA,posIdx)
        if (idxA==posIdx):
            print("same image: ",globals()['labels'][label])
            same_count+=1
        if(posIdx==None):
            print("None")

        # prepare a positive pair and update the images and labels
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label

        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]

        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    pairImages = np.array(pairImages)
    #print(pairImages.shape)
    #pairImages=pairImages.reshape((2, 480, 640, 1))
    pairLabels = np.array(pairLabels)
    #print(pairLabels.shape)
    #pairLabels=pairLabels.reshape((2, 480, 640, 1))
    print("number of pairs having same image: ",same_count)
    return (pairImages, pairLabels)

print("Training pairs:")
pairTrain, labelTrain = make_pairs(x_train, y_train)
print("Validation pairs:")
pairVal, labelVal = make_pairs(x_val, y_val)
print("Testing pairs:")
pairTest, labelTest = make_pairs(x_test, y_test)


def model_config(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(300, activation='relu', kernel_initializer='he_uniform'))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: keras.backend.abs(
        tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    # model.add(Dense(200, activation='softmax'))
    # compile the model
    opt = keras.optimizers.Adam(lr=0.001)
    siamese_net.compile(optimizer=opt, loss='binary_crossentropy',
                        metrics=['accuracy'])
    return siamese_net


def train_model(datax, datay, valx, valy):
    model = model_config(IMG_SHAPE)
    # fit model
    history = model.fit([datax[:, 0], datax[:, 1]], datay[:], epochs=30, batch_size=32,
                        validation_data=([valx[:, 0], valx[:, 1]], valy), verbose=2)
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
plt.savefig('../../Outputs/verification_model3_DA_stratify_graph.png')
plt.close()
model.save('../Models/human_2D_verification_model3_DA_stratify.h5')
print("saved")

# evaluation
print("Model evaluation on test dataset: ")
pred_prob = model.predict([pairTest[:, 0], pairTest[:, 1]])
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
plt.title('ROC curve on UND 2D Dataset')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend(['No skill', 'ROC Curve'], loc='lower right')
plt.savefig('../../Outputs/verification_model3_DA_stratify_roc.png')
plt.close()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
roc_display.plot()
plt.savefig('../../Outputs/verification_model3_DA_stratify_display_roc.png')
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

plt.savefig('../../Outputs/verification_model3_DA_stratify_confusionMatrix.png')
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
plt.savefig('../../Outputs/verification_model3_DA_stratify_EER.png')
plt.close()