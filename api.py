from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse 
from PIL import Image 
import numpy as np 
import tensorflow as tf 
import io
app=FastAPI()
model=tf.keras.models.load_model("waste_classifier_model.h5")
class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/",StaticFiles(directory="static",html=True))
@app.get("/") #main root
def root():
    return {"message":"Smart waste management system"}
@app.post("/predict")
async def predict(file:UploadFile=File(...)): #nonblocking i/o
    try:
        contents=await file.read()  #image is read in raw byte
        img=Image.open(io.BytesIO(contents)).convert("RGB") #wraps bytes into stream so PIL can convert to RGB
        img=img.resize((224,224))
        img_array=np.array(img)/255.0
        img_array=np.expand_dims(img_array,axis=0)
        prediction=model.predict(img_array)
        class_idx=int(np.argmax(prediction))
        confidence=float(prediction[0] [class_idx])*100 
        return JSONResponse(content={
            "predicted_class":class_names[class_idx],
            "confidence":f"{confidence:.2f}%"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))