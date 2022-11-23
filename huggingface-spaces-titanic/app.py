import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked):
    input_list = []
    input_list.append(Pclass)
    input_list.append(Sex)
    input_list.append(Age)
    input_list.append(SibSp)
    input_list.append(Parch)
    input_list.append(Fare)
    input_list.append(Embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    survival_url = "https://raw.githubusercontent.com/Yasaman97/ID2223-Titanic-Yasaman/main/img/" + str(res[0]) + ".jpg"
    img = Image.open(requests.get(survival_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Analytics",
    description="Experiment with Passenger class, Sex, Age, SibSp, Parch, Fare, and Embarked to predict Survival.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Pclass (int 1-3)"),
        gr.inputs.Number(default=1.0, label="Sex (1: female, 0: male)"),
        gr.inputs.Number(default=1.0, label="Age (float (in the dataset: 1-80))"),
        gr.inputs.Number(default=1.0, label="SibSp (int 0-3)"),
        gr.inputs.Number(default=1.0, label="Parch (int 0-4)"),
        gr.inputs.Number(default=1.0, label="Fare (float (in the dataset: 0-512)"),
        gr.inputs.Number(default=1.0, label="Embarked (int 0-2)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

