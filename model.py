from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(object):
    def __init__(self, config):
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.test_dir = config.test_dir
        self.num_train = config.num_train
        self.num_val = config.num_val
        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.image_size = config.image_size
        self.cnum = config.cnum
    
    def model_FER (self):
        """
        Feedforward model
        """
        model_cnn = Sequential()

        model_cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.image_size,self.image_size,1)))
        model_cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
        model_cnn.add(Dropout(0.25))

        model_cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
        model_cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
        model_cnn.add(Dropout(0.25))

        model_cnn.add(Flatten())
        model_cnn.add(Dense(1024, activation='relu'))
        model_cnn.add(Dropout(0.5))
        model_cnn.add(Dense(self.cnum, activation='softmax'))
        return model_cnn
    
    def data_preprocess(self):
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.image_size,self.image_size),
                batch_size=self.batch_size,
                color_mode="grayscale",
                class_mode='categorical')
        
        validation_generator = val_datagen.flow_from_directory(
                self.val_dir,
                target_size=(self.image_size,self.image_size),
                batch_size=self.batch_size,
                color_mode="grayscale",
                class_mode='categorical')
        return train_generator, validation_generator
    
    
