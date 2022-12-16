# yolov3-3d
An implementation of the 3-dimensional version of yolov3, using tensorflow 2

This repository is adapted from the following implementation by YunYang1994: https://github.com/YunYang1994/tensorflow-yolov3

This project uses an example of detecting liver lesions in 3d abdominal CT images.

To adapt this project for your own uses:
- Change data/dataset/liver_train.txt, data/dataset/liver_test.txt into your own training and testing annotations. The format of annotations is similar to the original repository: filepath y1,x1,z1,y2,x2,z2,class y1,x1,z1,y2,x2,z2,class
- Change how the data is loaded, in dataset.py, function parse_annotation(), line 233. This example uses .npy files, so the code to load the image is np.load(). 
- Change data/classes/liver.names to define your classes. 

This code is also rewritten using the functional API enabling eager execution, making it more compatible with tf2 as compared to the original repository. 
