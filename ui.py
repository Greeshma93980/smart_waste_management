import gradio as gr 
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
model=tf.keras.models.load_model("waste_classifier_model.h5")
class_names=['cardboard','glass','metal','paper','plastic','trash']
def predict_wastage(img):
    img=img.resize((224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)

    #predict
    prediction=model.predict(img_array)[0]
    prediction_class=class_names[np.argmax(prediction)]
    confidence=float(np.max(prediction))*100
    probs={cls:float(prob) for cls,prob in zip(class_names,prediction)}
    return prediction_class,confidence,probs
    return "hai"
with gr.Blocks() as demo:
    gr.Markdown("Waste Classification")
    with gr.Row():
        img_input=gr.Image(type="pil",label="upload an image")
        with gr.Column():
            out_class=gr.Textbox(label="predicted class")
            out_conf=gr.Number(label="confidence level")
            out_probs=gr.Label(label="class probability")
    btn=gr.Button("classify")
    btn.click(fn=predict_wastage,inputs=img_input,outputs=[out_class,out_conf,out_probs])
demo.launch()