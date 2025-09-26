---
layout: page
title: Age Estimation and Gender Classification
description: ML, TensorFlow, Python, Google Colab
img: assets/img/utkface.jpg
importance: 1
category: Personal
related_publications: true
---

In this project a Convolutional Neural Netowrk (CNN) model was trained to predict the age and the gender of a person using the functional API Keras.

We used a subsample of 500 images of the UTKFace dataset as our training set. However, it is possible to modify and use different datasets.

To create the CNN we used the functional API Keras version 3 that was integrated with Tensorflow.

Now we are going to create a step by step solution:

1. The first step is to import all the libraries that we are going to use.

```python
from google.colab import drive
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import tensorflow as tf
import keras
import numpy as np
```

2. If you are running using Google Colab set the environmental variable to `google_colab` otherwise to `local`

```python
env = 'google_colab' # change to local
```


3. Data processing

- Split the data into training and validations sets. 80% for training and 20% for validation

```python
all_image_files = [file for file in os.listdir(folder) if file.lower().endswith(('.jpg'))]

# Shuffle the dataset to ensure random distribution
random.seed(0)  # Ensure reproducibility
random.shuffle(all_image_files)

# Calculate the number of images for each set
n_train_val = len(all_image_files)
train_end = int(n_train_val * 0.8)

# Split the dataset
train_image_files = all_image_files[:train_end]
val_image_files = all_image_files[train_end:]
```

- Loading image data, gender labels and age values and normalize pixel values to the range [0, 1]

```python
def load_imgs_lables(dataset_path,filenames):
  print('load all image data, age and gender labels...')
  images = []
  age_labels = []
  gender_labels = []
  for current_file_name in filenames:
    img = cv2.imread(os.path.join(dataset_path, current_file_name))
    img = img / 255.0  # Normalize pixel values
    labels = current_file_name.split('_')
    age_label = int(labels[0])
    gender_label = int(labels[1])
    age_labels.append(age_label)
    gender_labels.append(gender_label)
    images.append(img)

# load data from the training set
train_images, train_age, train_gender = load_imgs_lables(folder, train_image_files)

# load data from the validation set
val_images, val_age, val_gender = load_imgs_lables(folder, val_image_files)
```

- Divide into batches to optimize training

```python
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, {'gender_out': train_gender, 'age_out': train_age}))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, {'gender_out': val_gender, 'age_out': val_age}))
val_dataset = val_dataset.batch(batch_size)
```
- Create and implemented data augmentation layer to flip images and create more samples 

```python
data_augmentation_layers = [
    tf.keras.layers.RandomFlip("horizontal")
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

```

6. Building CNN.
- First we declare the input shape and applied data augmentation as a new layer.

```python
input_shape = (128, 128, 3)
inputs = Input(shape=input_shape)
aug = data_augmentation(inputs)
```

- Second we build a shared convolutional block that learns low-level features for both age and gender.  It consists of two convolutional layers with filter sizes of 32 and 64. ReLU is used as the activation function for all layers in convolutions. After each convolutional layer, batch normalisation is added to prevent overfitting, followed by a max-pooling layer to reduce dimensionality. 

```python
conv_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(aug)
conv_1 = BatchNormalization()(conv_1)
conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_1)
conv_1 = BatchNormalization()(conv_1)
conv_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
```
- Third, after this block the model is divided into two branches. One for age and other for gender. 

The gender branch consists of three convolutional blocks. Each block contains a
convolutional, batch normalisation, and max-pooling layer. Filter sizes increase from 64
by a multiple of 2 in each layer. Additionally, L2 regularisation was applied to each
convolutional layer to mitigate overfitting, with the parameter value 0.001. This branch uses the sigmoid activation function to determine predictions.

```python
# age
conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_1)
conv_2 = BatchNormalization()(conv_2)
conv_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_2)
conv_2 = BatchNormalization()(conv_2)
conv_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

conv_3 = Conv2D(128, (3, 3) , padding='same', activation='relu')(conv_2) #256
conv_3 = BatchNormalization()(conv_3)
conv_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

conv_3 = Conv2D(256, (3, 3) , padding='same', activation='relu')(conv_3) #256
conv_3 = BatchNormalization()(conv_3)
conv_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

conv_4 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_3) #512
conv_4 = BatchNormalization()(conv_4)
conv_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten_age = Flatten() (conv_4)

# Fully Connected Head for Age Prediction
age_fc = Dense(128, activation='relu')(flatten_age)
age_fc = Dropout(0.30)(age_fc)
output_2 = Dense(1, activation='relu', name='age_out')(age_fc)

```

The age branch used five convolutional blocks, each block identical in structure to the
gender branch. Regularisation was not applied to age as it did not exhibit overfitting.
Filter sizes increased from 64 to 512, with one layer repeating filter sizes of 64. Finally, this branch usesReLU as the activation function to determine predictions.



```python
# gender
conv_2_g = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(conv_1)
conv_2_g = BatchNormalization()(conv_2_g)
conv_2_g = MaxPooling2D(pool_size=(2, 2))(conv_2_g)

conv_2_g = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(conv_2_g)
conv_2_g = BatchNormalization()(conv_2_g)
conv_2_g = MaxPooling2D(pool_size=(2, 2))(conv_2_g)

conv_2_g = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(conv_2_g)
conv_2_g = BatchNormalization()(conv_2_g)
conv_2_g = MaxPooling2D(pool_size=(2, 2))(conv_2_g)

flatten_gender = Flatten() (conv_2_g)

# Fully Connected Head for Gender Classification
gender_fc = Dense(128, activation='relu')(flatten_gender)
gender_fc = Dropout(0.30)(gender_fc)
output_1 = Dense(1, activation='sigmoid', name='gender_out')(gender_fc)

```
- Finally, we define the model

```python
modelA = keras.Model(inputs, [output_1, output_2])
```

7. Train the CNN model
For training, we selected the Adam optimizer, generally the learning rate goes from 0.01 to 0.0001, and we determinated that the the optimal value was 0.001. Lower learning rates
resulted in slower training without improved model performance. Higher values led to
unstable training with fluctuating results.

```python
opt = keras.optimizers.Adam(learning_rate = 0.001)
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                         restore_best_weights=True)
modelA.compile(loss={'gender_out': 'binary_crossentropy', 'age_out': 'mae'},
               optimizer=opt,
               metrics={'gender_out': 'accuracy', 'age_out': 'mae'})

# Train the model
epochs = 50
modelA_fit = modelA.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks = [callback]
)

```


8. Evaluate the model
To evaluate the model we plotted 4 learning curves. The loss of the gender classification over the training and validation set, The accuracy of the gender classification over the training and validation set, The loss of the age estimation over the training and validation set, and The MAE of the age estimation over the training and validation set.

We observe noise in age and gender losses because of the low learning rate, because the model performs larger steps to find optimal values.  We observe minor overfitting of gender and age, given the loss curves, mitigated by early stopping callbacks, returning the model weights to the epoch with the lowest validation loss.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/results_cnn.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Most gender accuracy improvement was observed in the first 10 epochs, plateauing afterward. A similar observation is seen in age in the first 20 epochs. Finally, we got an accuracy of 88% for gender and MAE of 6.95 years for age.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/results_2_cnn.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

To download all the code and the dataset visit the [repository](https://github.com/crist2201/cnn-utkface)
Every project has a beautiful feature showcase page.
It's easy to include images in a flexible 3-column grid format.
Make your photos 1/3, 2/3, or full width.

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/results_cnn.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images, even citations {% cite einstein1950meaning %}.
Say you wanted to write a bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %}
