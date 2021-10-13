import json
from PIL import Image
import os
import streamlit as st
import pandas as pd
import numpy as np

from resnet_model import Resnet_Model

@st.cache()
def load_model(path='../models/trained_model_resnet50.pt', device='cpu'):
    """Retrieves the trained model and maps it to the CPU by default, can also specify GPU here."""
    model = Resnet_Model(path_to_pretrained_model=path, map_location=device)
    return model

@st.cache()
def load_index_to_label_dict(path='../src/index_to_class_label.json'):
    """Retrieves and formats the index to class label lookup dictionary needed to 
    make sense of the predictions. When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict

@st.cache()
def predict(img, index_to_label_dict, model, k):
    """Transforming input image according to ImageNet paper
    The Resnet was initially trained on ImageNet dataset
    and because of the use of transfer learning, I froze all
    weights and only learned weights on the final layer.
    The weights of the first layer are still what was
    used in the ImageNet paper and we need to process
    the new images just like they did.
    
    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions."""
    formatted_predictions = model.predict_proba(img, k, index_to_label_dict)
    return formatted_predictions

if __name__ == '__main__':
    model = load_model()
    index_to_class_label_dict = load_index_to_label_dict()

    st.title('Bird Species Classification ')
    instructions = """
        Upload your own image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file:
        img = Image.open(file)
        prediction = predict(img, index_to_class_label_dict, model, 5)

        st.title("Here is the image you've selected")
        resized_image = img.resize((336,336))
        st.image(resized_image)
        st.title("Here are the five most likely bird species")
        df = pd.DataFrame(data=np.zeros((5, 2)), 
                        columns=['Species', 'Confidence Level'],
                        index=np.linspace(1, 5, 5, dtype=int))
        for idx, p in enumerate(prediction):
            link = 'https://en.wikipedia.org/wiki/' + p[0].lower().replace(' ', '_')
            df.iloc[idx, 0] = f'<a href="{link}" target="_blank">{p[0].title()}</a>'
            df.iloc[idx, 1] = p[1]
        st.write(df.to_html(escape=False), unsafe_allow_html=True)
