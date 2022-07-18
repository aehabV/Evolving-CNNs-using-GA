from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
from matplotlib import pyplot as plt
import sys


def load_dataset(train_dir, test_dir, batches, epochs):

    datagen = image.ImageDataGenerator(featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=135,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode="nearest")
    
    train_dataset = datagen.flow_from_directory(train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batches,
    shuffle=True,               
    class_mode="categorical" 
    )

    test_dataset = datagen.flow_from_directory(test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batches,
    shuffle=True,               
    class_mode="categorical"
    )
        
    train, test = ([] for l in range(2)) 
    trainPath = train_dataset.directory
    testPath = test_dataset.directory

    for i in tqdm (range(train_dataset.samples), desc="Loading training images", colour='#D1B37D'):
      imgPath = Image.open(trainPath+train_dataset.filenames[i])
      img = np.array(imgPath)
      train.append(img)  
      
    for i in tqdm (range(test_dataset.samples), desc="Loading tesing images", colour='#1482F7'):
      imgPath = Image.open(testPath+test_dataset.filenames[i])
      img = np.array(imgPath)
      test.append(img)    
    
    x_train = np.array(train)
    x_test = np.array(test)
    y_train = to_categorical(train_dataset.labels)
    y_test = to_categorical(test_dataset.labels)
    num_classes = train_dataset.num_classes
    
    print("Batch Size --> {}".format(batches))
    print("Num of Classes --> {}".format(num_classes))
    print("Num of Epochs --> {}".format(epochs))
    
    
    dataset = {
        'batch_size': batches,
        'num_classes': num_classes,
        'epochs': epochs,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return dataset

def save_network(network):
    object_file = open(network.name + '.obj', 'wb')
    pickle.dump(network, object_file)


def load_network(name):
    object_file = open(name + '.obj', 'rb')
    return pickle.load(object_file)


def order_indexes(self):
    i = 0
    for block in self.block_list:
        block.index = i
        i += 1


def plot_training(history):                                           # plot diagnostic learning curves
    plt.figure(figsize=[8, 6])											# loss curves
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_loss_plot.png')

    plt.figure(figsize=[8, 6])											# accuracy curves
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_acc_plot.png')
    plt.close()


def plot_statistics(stats):
    plt.figure(figsize=[8, 6])											# fitness curves
    plt.plot([s[0] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][0]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['BestFitness', 'InitialFitness'], fontsize=18)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('FitnessValue', fontsize=16)
    plt.title('Fitness Curve', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_fitness_plot.png')

    plt.figure(figsize=[8, 6])											# parameters curves
    plt.plot([s[1] for s in stats], 'r', linewidth=3.0)
    plt.plot([stats[0][1]] * len(stats), 'b', linewidth=3.0)
    plt.legend(['BestParamsNum', 'InitialParamsNum'], fontsize=18)
    plt.xlabel('Generations', fontsize=16)
    plt.ylabel('ParamsNum', fontsize=16)
    plt.title('Parameters Curve', fontsize=16)
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_params_plot.png')
    plt.close()
