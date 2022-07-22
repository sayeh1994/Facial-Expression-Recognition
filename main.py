import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.optimizers import Adam
# Checkpoint the weights when validation accuracy improves
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from model import CNN
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def str2bool(v):
    return v.lower() in ('true')

# plots accuracy and loss curves
def plot_model_history(model_history, directory, savefig = True):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    if savefig:
        fig.savefig(os.path.join(directory,'plot.png'))
    plt.show()

def label(files):
  y_label = []
  emotion_dict = {"angry": 0 , "disgusted": 1, "fearful": 2, "happy": 3, "sad": 4, "surprised": 5}
  emotion = ["angry" , "disgusted", "fearful", "happy", "sad", "surprised"]
  for f in files:
    if emotion[0] in f:
      y_label.append(emotion_dict[emotion[0]])
    elif emotion[1] in f:
      y_label.append(emotion_dict[emotion[1]])
    elif emotion[2] in f:
      y_label.append(emotion_dict[emotion[2]])
    elif emotion[3] in f:
      y_label.append(emotion_dict[emotion[3]])
    elif emotion[4] in f:
      y_label.append(emotion_dict[emotion[4]])
    elif emotion[5] in f:
      y_label.append(emotion_dict[emotion[5]])
  return y_label

def confusion_map(y_true, y_predict, annotex, directory, name, savefig = True):

    cm = confusion_matrix(y_true, y_predict)
    ConfusionMatrixDisplay(cm, display_labels=annotex).plot()
    if savefig:
        plt.savefig(os.path.join(directory, "confusion_matrix_"+name+".jpg"))
    return cm

def main(config):
    mode = config.mode
    num_train = config.num_train
    num_val = config.num_val
    batch_size = config.batch_size
    num_epoch = config.num_epoch
    stop_patient = config.stop_patient
    train_dir = config.train_dir
    val_dir = config.val_dir
    test_dir = config.test_dir
    cm_name = config.CM_name
    save = config.save_fig
    model_save_dir = config.model_save_dir
    result_save_dir = config.result_save_dir
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)
    cnn = CNN(config)
    
    # num_val = int(len(glob.glob(val_dir+"/*/*.jpg")))
    # num_train = int(len(glob.glob(train_dir+"/*/*.jpg")))
    # If you want to train the same model or try other models, go for this
    if mode == "train":
        model = cnn.model_FER()
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
        train_generator, validation_generator = cnn.data_preprocess()
        checkpoint = ModelCheckpoint(os.path.join(model_save_dir, "best_model"), save_best_only=True)
        stop_early = EarlyStopping(monitor='val_loss', patience=stop_patient)
        if stop_patient != 0:
            model_info = model.fit(
                train_generator,
                steps_per_epoch=num_train // batch_size,
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_val // batch_size,
                callbacks=[stop_early, checkpoint],
                )
        else:
            model_info = model.fit(
                train_generator,
                steps_per_epoch=num_train // batch_size,
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_val // batch_size,
                callbacks=[checkpoint],
                )
        plot_model_history(model_info,result_save_dir, savefig=save)
        model.save_weights(os.path.join(model_save_dir,'model.h5'))
    
    # emotions will be displayed on your face from the webcam feed
    elif mode == "test":
        files_test = glob.glob(test_dir+"/*/*.jpg")
        ylabel = label(files_test)
        model = cnn.model_FER()
        model.load_weights(os.path.join(model_save_dir, 'model.h5'))
        y_predict = []
        # dictionary which assigns each label an emotion (alphabetical order)
        # emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        for f in files_test:
          frame = cv2.imread(f)
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (64, 64)), -1), 0)
          prediction = model.predict(cropped_img)
          maxindex = int(np.argmax(prediction))
          y_predict.append(maxindex)
         
        print("Test accuracy is:{}".format(accuracy_score(ylabel, y_predict)))
        emotion = ["angry" , "disgusted", "fearful", "happy", "sad", "surprised"]
        cm = confusion_map(ylabel, y_predict, emotion, result_save_dir, cm_name, savefig=save)
        print(classification_report(ylabel, y_predict, labels=[0,1,2,3,4,5]))
        ConfusionMatrixDisplay(cm, display_labels=emotion).plot()
    

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=['train', 'test', 'display'])
    ap.add_argument("--train_dir",default="./data/train")
    ap.add_argument("--test_dir",default="./data/test")
    ap.add_argument("--val_dir",default="./data/val")
    ap.add_argument("--num_train", type=int,default=28000)
    ap.add_argument("--num_val", type=int,default=7000)
    ap.add_argument("--batch_size", type=int,default=64)
    ap.add_argument("--num_epoch", type=int,default=50)
    ap.add_argument("--image_size", type=int,default=64)
    ap.add_argument("--cnum", type=int,default=7)
    ap.add_argument("--save_fig", type=str2bool,default=True)
    ap.add_argument("--model_save_dir",default="./emotion_detection/model")
    ap.add_argument("--result_save_dir",default="./emotion_detection/result")
    ap.add_argument("--stop_patient", type=int,default=5)
    ap.add_argument("--CM_name",default="CM")
    config = ap.parse_args()
    main(config)
