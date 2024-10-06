import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from time import time

# Define model paths
model_paths = {
    "ResNet50": "your file locatio",
    "VGG16": "your file locatio",
    "Custom": "your file locatio"
}

# Load models
models = {}
for model_name, model_path in model_paths.items():
    models[model_name] = load_model(model_path)

# List of breeds for prediction
new_list = ['yorkshire_terrier', 'whippet', 'welsh_springer_spaniel', 'walker_hound', 'toy_terrier', 'tibetan_terrier', 'sussex_spaniel', 'standard_poodle', 'soft-coated_wheaten_terrier', 'siberian_husky', 'shetland_sheepdog', 'scottish_deerhound', 'schipperke', 'saluki', 'rottweiler', 'redbone', 'pomeranian', 'pekinese', 'otterhound', 'norwich_terrier', 'norfolk_terrier', 'miniature_schnauzer', 'miniature_pinscher', 'maltese_dog', 'malamute', 'leonberg', 'labrador_retriever', 'komondor', 'kelpie', 'japanese_spaniel', 'irish_wolfhound', 'irish_terrier', 'ibizan_hound', 'greater_swiss_mountain_dog', 'great_dane', 'golden_retriever', 'german_short-haired_pointer', 'french_bulldog', 'eskimo_dog', 'english_springer', 'english_foxhound', 'dingo', 'dandie_dinmont', 'collie', 'clumber', 'chihuahua', 'cardigan', 'bull_mastiff', 'briard', 'boxer', 'boston_bull', 'border_terrier', 'bluetick', 'blenheim_spaniel', 'bernese_mountain_dog', 'beagle', 'basenji', 'appenzeller', 'airedale', 'afghan_hound']

breed_map = {
    0: 'affenpinscher', 1: 'afghan_hound', 2: 'african_hunting_dog', 3: 'airedale',
    4: 'american_staffordshire_terrier', 5: 'appenzeller', 6: 'australian_terrier',
    7: 'basenji', 8: 'basset', 9: 'beagle', 10: 'bedlington_terrier', 11: 'bernese_mountain_dog',
    12: 'black-and-tan_coonhound', 13: 'blenheim_spaniel', 14: 'bloodhound', 15: 'bluetick',
    16: 'border_collie', 17: 'border_terrier', 18: 'borzoi', 19: 'boston_bull',
    20: 'bouvier_des_flandres', 21: 'boxer', 22: 'brabancon_griffon', 23: 'briard',
    24: 'brittany_spaniel', 25: 'bull_mastiff', 26: 'cairn', 27: 'cardigan',
    28: 'chesapeake_bay_retriever', 29: 'chihuahua', 30: 'chow', 31: 'clumber',
    32: 'cocker_spaniel', 33: 'collie', 34: 'curly-coated_retriever', 35: 'dandie_dinmont',
    36: 'dhole', 37: 'dingo', 38: 'doberman', 39: 'english_foxhound', 40: 'english_setter',
    41: 'english_springer', 42: 'entlebucher', 43: 'eskimo_dog', 44: 'flat-coated_retriever',
    45: 'french_bulldog', 46: 'german_shepherd', 47: 'german_short-haired_pointer',
    48: 'giant_schnauzer', 49: 'golden_retriever', 50: 'gordon_setter', 51: 'great_dane',
    52: 'great_pyrenees', 53: 'greater_swiss_mountain_dog', 54: 'groenendael', 55: 'ibizan_hound',
    56: 'irish_setter', 57: 'irish_terrier', 58: 'irish_water_spaniel', 59: 'irish_wolfhound',
    60: 'italian_greyhound', 61: 'japanese_spaniel', 62: 'keeshond', 63: 'kelpie',
    64: 'kerry_blue_terrier', 65: 'komondor', 66: 'kuvasz', 67: 'labrador_retriever',
    68: 'lakeland_terrier', 69: 'leonberg', 70: 'lhasa', 71: 'malamute', 72: 'malinois',
    73: 'maltese_dog', 74: 'mexican_hairless', 75: 'miniature_pinscher', 76: 'miniature_poodle',
    77: 'miniature_schnauzer', 78: 'newfoundland', 79: 'norfolk_terrier', 80: 'norwegian_elkhound',
    81: 'norwich_terrier', 82: 'old_english_sheepdog', 83: 'otterhound', 84: 'papillon',
    85: 'pekinese', 86: 'pembroke', 87: 'pomeranian', 88: 'pug', 89: 'redbone',
    90: 'rhodesian_ridgeback', 91: 'rottweiler', 92: 'saint_bernard', 93: 'saluki',
    94: 'samoyed', 95: 'schipperke', 96: 'scotch_terrier', 97: 'scottish_deerhound',
    98: 'sealyham_terrier', 99: 'shetland_sheepdog', 100: 'shih-tzu', 101: 'siberian_husky',
    102: 'silky_terrier', 103: 'soft-coated_wheaten_terrier', 104: 'staffordshire_bullterrier',
    105: 'standard_poodle', 106: 'standard_schnauzer', 107: 'sussex_spaniel', 108: 'tibetan_mastiff',
    109: 'tibetan_terrier', 110: 'toy_poodle', 111: 'toy_terrier', 112: 'vizsla', 113: 'walker_hound',
    114: 'weimaraner', 115: 'welsh_springer_spaniel', 116: 'west_highland_white_terrier', 117: 'whippet',
    118: 'wire-haired_fox_terrier', 119: 'yorkshire_terrier'
}

# Function to preprocess the image
def preprocess_image(image, target_size):
    img = cv2.resize(image, target_size)
    img_array = preprocess_input(np.expand_dims(img[..., ::-1].astype(np.float32), axis=0))
    return img_array

# Function to predict the breed
def predict_breed(image, model_name):
    start_time = time()
    img_array = preprocess_image(image, (224, 224))
    pred_val = models[model_name].predict(img_array)
    end_time = time()
    
    if model_name in ["ResNet50", "VGG16"]:
        # Use new_list for ResNet50 and VGG16 models
        predicted_breed = sorted(new_list)[np.argmax(pred_val)]
    else:
        # Use labels_df for other models
        breed_index = np.argmax(pred_val)
        predicted_breed = breed_map.get(breed_index, "Unknown")
    
    prediction_time = end_time - start_time
    return predicted_breed, prediction_time


# Function to display model architecture
def display_model_architecture(model_name):
    model = models[model_name]
    dot_img_file = f'{model_name}_model.png'
    plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
    st.image(dot_img_file)

# Streamlit app
st.sidebar.title('Options')
option = st.sidebar.selectbox('Select Option', ['Home', 'Model Architecture', 'Prediction'])

if option == 'Home':
    left_bone_emoji = "&#x1F9B4;"  # This is the bone emoji
    finger_emoji = "&#x1F449;"  # This is the finger pointing right emoji

    st.markdown(f"<h1 style='text-align: center;'><span>{left_bone_emoji}</span> Breed Finder <span>{left_bone_emoji}</span></h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: red;'>" + finger_emoji + " Introduction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 20px; color: white;'>Identifying dog breeds accurately is vital for various applications including veterinary health, animal welfare, and legal compliance. Traditional identification methods require expert knowledge and are not scalable. This project aims to overcome these limitations by developing an automated system capable of identifying dog breeds from photos using complex model-based approaches.</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='color: orange; text-align: left;'>" + finger_emoji + " Model Development</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;'>● Custom Model: Designed specifically for this project, this model employs convolutional neural layers tailored to capture distinctive features across various image conditions.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;'>● VGG16 and ResNet50: These established models were adapted by altering their final layers to focus on breed identification, chosen for their proven effectiveness in feature extraction.</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: left; color: blue;'>" + finger_emoji + " Conclusion</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;'>This project has successfully devised a breed identification system, with ResNet50 outperforming other models. The bespoke model, though less precise, provided crucial insights into specialized feature design. Challenges included handling unbalanced data and managing the computational demands of complex models. Future initiatives will focus on integrating more sophisticated combination techniques and enlarging the dataset to encompass less common breeds, aiming for a universally applicable system.</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='color: green;'>" + finger_emoji + " Work Submitted by</h2>", unsafe_allow_html=True)
    st.markdown('* Zangetsu')

elif option == 'Model Architecture':
    st.title('Model Architecture')
    selected_model = st.selectbox('Select Model', list(models.keys()))
    display_model_architecture(selected_model)

elif option == 'Prediction':
    st.title('Dog Breed Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    model_name = st.selectbox("Select Model", list(model_paths.keys()))

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            predicted_breed, prediction_time = predict_breed(image, model_name)
            st.write("Predicted Breed:", predicted_breed)
            st.write("Prediction Time:", round(prediction_time, 2), "seconds")