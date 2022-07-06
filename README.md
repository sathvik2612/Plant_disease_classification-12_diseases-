# AGRi App( predicts diseases from three plants )

Farmers in growing countries like India, Bangladesh and some poor countries doesn't have knowledge on diseases of plants. Survey's show that more than 40 percent of soil infertility is because using unneccesary fertilizers and pesticides. 

If farmers come to know what type of disease is present then farmer can search in google and get the right fertilizer or pesticide. 

That's why I started this project.

This model can be used for even 100's of diseases of 100 different plants. As datasets of all disease are not available I have done only for 3 plants. 

Building App using React Native and Tensorflow. Anyone Intrested can contact me for collaboration at 20bec024@iiitdwd.ac.in

## Diseases

This can predict 

Pepper Bell Bacterial.

Potato Early Blight.

Potato Late Blight.

Tomato Bacterial Spot.

Tomato Early Blight.

Tomato Late Blight.

Tomato Leaf Mold.

Tomato Septoria Leaf Soft.

Tomato Spider Mites Two Spotted Spider Mite.

Tomato Target Spot.

Tomato Yellow Leaf Curl Virus.

Tomato Mosaic Virus.

##Model

This model is built with Convolutional Neural Networks with Tensorflow.

![CNN](https://user-images.githubusercontent.com/82766969/177565715-ffa4db19-00b7-46fe-a50e-2b0e4a5e55f8.jpg)

```
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)
```










