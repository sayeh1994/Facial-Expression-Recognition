# Facial-Expression-Recognition

The goal of this project was to improve the performance of facial expression recognition model by augmenting the training dataset with generated data.

> the process of generating synthetic dataset is presented in another repository [synthetic facial expression](https://github.com/sayeh1994/synthesizin_facial_expression.git).

**Please keep in mind to use the label of your training classes as ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']**

To train the model please run the code below:

```bash
python main.py --mode="train" --train_dir="./train"  --val_dir="./validation" --model_save_dir="./model" --result_save_dir="./result" --num_train=600 --num_val=60 --batch_size=16 --num_epoch=50 --image_size=64 --cnum=6 --stop_patient=5
```

`cnum` is the number of classes you want to do the recognition. The image dimension is 64x64. `num_train` and `num_val` can be adjusted based on the total number of images in your training and validation set.

The model architecture is as follow:

![The 3d model architecture](https://github.com/sayeh1994/Facial-Expression-Recognition/blob/main/images/Model-3d-architecture.jpg)

For testing the trained model, you can run the following command:

```bash
python main.py --mode="test" --test_dir="./test" --CM_name="test" --model_save_dir="./model"  --result_save_dir="./result" --cnum=6 --image_size=64
```

You can find the pre-trained model for different augmentation in [MyDrive](https://drive.google.com/drive/folders/1gvf4V1Td0gdsmHEOeMkxg9WmqlqsMLvW?usp=sharing).
