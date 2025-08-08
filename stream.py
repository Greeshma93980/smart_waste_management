import streamlit as st
import tensorflow as tf 
import cv2 
import numpy as np
from PIL import Image
model=tf.keras.models.load_model("waste_classifier_model.h5")
class_names= ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] 
st.title("Smart Waste Management System")
st.subheader("upload an image to classify its category")
uploaded_file=st.file_uploader("Choose an image..",type=["png","jpg","jpeg"])
if uploaded_file:
    img=Image.open(uploaded_file).convert("RGB")  #opens image in streamlit
    st.image(img,caption="Uploaded image",use_container_width=True) # to fit according to screen width
    #preprocess 
    #uploading images into model
    img=img.resize((224,224)) 
    img_array=np.array(img) #converting PIL to numpy aray
    img_array=np.expand_dims(img_array,axis=0) #batch dimension,model take input in this form only
    img_array=img_array/255.0 #normalize pixels 255 to 0 and 1
    #predict
    prediction=model.predict(img_array)
    class_idx=np.argmax(prediction) # highest of index probablitity
    confidence=prediction[0][class_idx]*100
    st.success(f"Predicted: {class_names[class_idx]} ({confidence:.2f}% confidence)")