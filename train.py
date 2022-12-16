import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from common import *
from backbone import *
from utils import *
from yolov3 import YoloV3
from config import cfg
import time
from dataset import Dataset

LEARNING_RATE = 0.0001
EPOCHS = 50
TOTAL_EPOCHS = 50

weights_folder = r'/home/GPU/Arthur/Code/yolo/yolov3_3d/weights/yolov3_3d_basic'

checkpoint_path = os.path.join(weights_folder,
                               'YoloV3_training_{}_{}.h5'.format(
                                   TOTAL_EPOCHS-EPOCHS+1, TOTAL_EPOCHS))
checkpoint_path_best = os.path.join(weights_folder,
                               'YoloV3_training_{}_{}_best.h5'.format(
                                   TOTAL_EPOCHS-EPOCHS+1, TOTAL_EPOCHS))
model_save_path = os.path.join(weights_folder,
                               'YoloV3_{}epoch.h5'.format(TOTAL_EPOCHS)
                               )
csv_log_path = os.path.join(weights_folder,
                            'YoloV3_training_{}epoch.log'.format(TOTAL_EPOCHS)
                            )
prev_model_path = os.path.join(weights_folder,
                               "YoloV3_{}epoch.h5".format(TOTAL_EPOCHS-EPOCHS)
                               )

train_dataset = Dataset('train')
val_dataset = Dataset('test')

train_steps = len(train_dataset)
val_steps = len(val_dataset)

model = YoloV3()

model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
  metrics = []
  )

# Load model if available
if TOTAL_EPOCHS - EPOCHS != 0:
    model.load_weights(prev_model_path)
    print("\n\nLoaded model from epoch {}\n\n".format(TOTAL_EPOCHS-EPOCHS))
else:
    print("\n\nTraining new model.\n\n")
    

# Create checkpoint to save model after every epoch
cp_callback_best = ModelCheckpoint(
    filepath=checkpoint_path_best,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False
)

# Create checkpoint to save metrics
csv_logger = CSVLogger(csv_log_path)

history = model.fit(train_dataset,
		    epochs = EPOCHS,
		    validation_data = val_dataset,
		    steps_per_epoch = train_steps,
		    validation_steps = val_steps,
		    callbacks = [cp_callback, cp_callback_best, csv_logger]
		    )

model.save_weights(model_save_path)

print("Model finished training at {}".format(time.ctime()))