#Machine learning related code
import os
import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
from tensorflow.keras.applications import MobileNetV2

base_dir="archive (12)/Garbage classification/Garbage classification"
print("classes available ",os.listdir(base_dir))

train_ds=tf.keras.utils.image_dataset_from_directory(base_dir,validation_split=0.2,subset="training",image_size=(224,224),batch_size=32,seed=123)
test_ds=tf.keras.utils.image_dataset_from_directory(base_dir,validation_split=0.2,subset="validation",image_size=(224,224),batch_size=32,seed=123)

class_names=train_ds.class_names
print("class names",class_names)
image_path=f"{base_dir}/glass/glass126.jpg"
img=cv2.imread(image_path)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()

#epoches => 25 iterations
#Auto tuning to execute fast

AUTOTUNE=tf.data.AUTOTUNE
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds=test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#categorize of classes
classes=os.listdir(base_dir)
num_classes= len(classes)
print(f"we have{num_classes} classes")
tot_images=0
for c in classes:
    images=os.listdir(os.path.join(base_dir,c))
    tot_images+=len(images)
    print(f"for category {c},we have {len(images)} images")
base_model=MobileNetV2(input_shape=(224,224)+(3,),include_top=False,weights="imagenet")
base_model.trainable=False  #freeze base model
models=models.Sequential([base_model,layers.GlobalAveragePooling2D(),layers.Dense(128,activation="relu"),layers.Dropout(0.3),layers.Dense(6,activation="softmax")])
#compile th model
models.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#run the model 
pred=models.fit(train_ds,validation_data=test_ds,epochs=25)
models.save("waste_classifier_model.h5")