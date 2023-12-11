import tensorflow as tf


layers = tf.keras.layers
Adadelta = tf.keras.optimizers.Adadelta
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
from datetime import datetime

labels_nums = 5
batch_size = 16
resize_height = 224
resize_width = 224
depths = 3
data_shape = (batch_size, resize_height, resize_width, depths)

input_images = tf.keras.layers.Input(shape=data_shape[1:], name='input')
input_labels = tf.keras.layers.Input(shape=(labels_nums,), name='label')
keep_prob = tf.keras.layers.Input(tf.float32, name='keep_prob')
is_training = tf.keras.layers.Input(tf.bool, name='is_training')

# Define the model using Keras
base_model = tf.keras.applications.MobileNetV2(input_shape=data_shape[1:], include_top=False, weights=None)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.8)(x)
output = layers.Dense(labels_nums, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adadelta(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

