import streamlit as st
from main import predict  # Assuming 'predict' is a function in main.py that takes text input and returns emotion predictions

# Sample function to predict emotion (replace with your model)
emotions = ['anger','fear','joy','sadness',	'surprise']
emojis = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'sadness': '😢',
    'surprise': '😲',
}
    
intensity_colors ={
    0: '#D3D3D3',  # Neutral
    1: '#FFFF66',
    2: '#FFA500',
    3: '#FF0000'   # High intensity
}
    
intensity_labels = ["None", "Low", "Moderate", "High"]

    
def display_emotion(output):
    st.subheader("🔍 Predicted Emotion Intensity")
    
    output = output[0]  # Assuming output is a list of scores for each emotion
    
    for idx, intensity in enumerate(output):
        emotion = emotions[idx]
        emoji = emojis[emotion]
        color = intensity_colors[intensity]
        label = intensity_labels[intensity]
        
        if label == "None":
            continue
        st.markdown(f"""
        <div style='
            background-color:{color};
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:20px;
            margin-bottom:15px;
        '>
        {emoji} <br><b>{emotion.upper()}</b><br>Intensity: <b>{label}</b>
        </div>
        """, unsafe_allow_html=True)
    
    

st.title("Find Your Emotion")

user_input = st.text_input("Enter your text:")

if st.button("Predict Emotion"):
    if user_input:
        emotion = predict(user_input)
        output = display_emotion(emotion)
        st.write(f"**Predicted Emotion:** {emotion}")
    else:
        st.write("Please enter some text to predict the emotion.")
