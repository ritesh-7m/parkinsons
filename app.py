import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Parkinson's Disease MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .healthy-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .disease-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .model-info {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        color: #0056b3;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Model loading functions
@st.cache_resource
def load_trained_model(model_path):
    """Load a trained model from file"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")
        return None

@st.cache_data
def load_class_indices():
    """Load class indices if available"""
    try:
        if os.path.exists('class_indices.json'):
            with open('class_indices.json', 'r') as f:
                return json.load(f)
        else:
            # Default mapping based on your notebook
            return {'healthy': 0, 'parkinsons': 1}
    except:
        return {'healthy': 0, 'parkinsons': 1}

def create_demo_model():
    """Create a demonstration model if trained models are not available"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Initialize with random weights
    dummy_input = np.random.random((1, 224, 224, 3))
    model.predict(dummy_input, verbose=0)
    
    return model

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize pixel values (same as your training: rescale=1./255)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_parkinsons(model, processed_image, class_indices):
    """Make prediction using the model"""
    prediction = model.predict(processed_image, verbose=0)
    confidence = float(prediction[0][0])
    
    # Based on your notebook's binary classification setup
    # class_mode='binary' means: 0 = first class alphabetically, 1 = second class
    # So 'healthy' = 0, 'parkinsons' = 1
    
    if confidence > 0.5:
        return "Parkinson's Disease Detected", confidence
    else:
        return "Healthy", 1 - confidence

# App header
st.markdown('<h1 class="main-header">🧠 Parkinson\'s Disease MRI Classifier</h1>', unsafe_allow_html=True)

# Warning disclaimer
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
    It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals 
    for medical advice and diagnosis.
</div>
""", unsafe_allow_html=True)

# Model selection
st.sidebar.header("🤖 Model Selection")

# Check for available trained models
available_models = {}
model_files = {
    'VGG16': 'parkinsons_vgg16_model.h5',
    'ResNet50': 'parkinsons_resnet50_model.h5',
    'InceptionV3': 'parkinsons_inceptionv3_model.h5'
}

for model_name, model_file in model_files.items():
    if os.path.exists(model_file):
        available_models[model_name] = model_file

if available_models:
    selected_model_name = st.sidebar.selectbox(
        "Choose a trained model:",
        list(available_models.keys())
    )
    
    st.sidebar.markdown(f"""
    <div class="model-info">
        <strong>Selected Model:</strong> {selected_model_name}<br>
        <strong>File:</strong> {available_models[selected_model_name]}<br>
        <strong>Status:</strong> ✅ Trained Model Available
    </div>
    """, unsafe_allow_html=True)
    
    # Load the selected model
    model = load_trained_model(available_models[selected_model_name])
else:
    st.sidebar.warning("No trained models found. Using demonstration model.")
    selected_model_name = "Demo CNN"
    model = create_demo_model()
    
    st.sidebar.markdown("""
    <div class="model-info">
        <strong>Model Status:</strong> Using demonstration model<br>
        <strong>Note:</strong> Place your trained .h5 files in the app directory for better accuracy
    </div>
    """, unsafe_allow_html=True)

# Load class indices
class_indices = load_class_indices()

# Information about the app
with st.expander("ℹ️ About this Application"):
    st.write(f"""
    This application uses deep learning to analyze MRI brain images for potential signs of Parkinson's disease.
    
    **Current Model:** {selected_model_name}
    
    **How it works:**
    1. Upload an MRI brain image (JPEG, PNG, or other common formats)
    2. The AI model analyzes the image features
    3. Get a classification result with confidence score
    
    **Training Details:**
    - **Image Size:** 224x224 pixels
    - **Data Augmentation:** Rotation, shift, shear, zoom, horizontal flip
    - **Architecture:** Transfer learning with pre-trained ImageNet weights
    - **Classification:** Binary (Healthy vs Parkinson's)
    
    **Dataset Information:**
    The models were trained on MRI brain images with the following structure:
    - Healthy brain MRI images
    - Parkinson's disease brain MRI images
    - Image preprocessing: Rescaling to [0,1] range
    """)

if model is None:
    st.error("❌ Failed to load model. Please check your model files.")
    st.stop()

st.success(f"✅ {selected_model_name} model loaded successfully!")

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload MRI Brain Image",
    type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
    help="Upload a brain MRI image for Parkinson's disease classification"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original MRI Image", use_column_width=True)
        
        # Display image info
        st.write(f"**Image Size:** {image.size}")
        st.write(f"**Image Mode:** {image.mode}")
        st.write(f"**File Size:** {uploaded_file.size} bytes")
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        try:
            # Preprocess the image
            with st.spinner("Preprocessing image..."):
                processed_image = preprocess_image(image)
            
            # Make prediction
            with st.spinner(f"Analyzing with {selected_model_name}..."):
                result, confidence = predict_parkinsons(model, processed_image, class_indices)
            
            # Display results
            if "Parkinson's" in result:
                st.markdown(f"""
                <div class="result-box disease-result">
                    <h3>🔍 Classification Result</h3>
                    <h2>{result}</h2>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box healthy-result">
                    <h3>🔍 Classification Result</h3>
                    <h2>{result}</h2>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional information
            st.subheader("📊 Detailed Analysis")
            
            # Create a confidence bar chart
            fig, ax = plt.subplots(figsize=(8, 2))
            
            if "Parkinson's" in result:
                colors = ['#dc3545', '#28a745']
                values = [confidence, 1-confidence]
                labels = ['Parkinson\'s', 'Healthy']
            else:
                colors = ['#28a745', '#dc3545']
                values = [confidence, 1-confidence]
                labels = ['Healthy', 'Parkinson\'s']
            
            bars = ax.barh(labels, values, color=colors, alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence Score')
            ax.set_title(f'{selected_model_name} Classification Confidence')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(value/2, bar.get_y() + bar.get_height()/2, 
                       f'{value:.2%}', ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Model-specific information
            st.subheader("🤖 Model Information")
            if selected_model_name == "VGG16":
                st.write("""
                **VGG16 Architecture:**
                - 16 layers deep convolutional network
                - Pre-trained on ImageNet
                - Transfer learning with frozen early layers
                - Fine-tuned for medical image classification
                """)
            elif selected_model_name == "ResNet50":
                st.write("""
                **ResNet50 Architecture:**
                - 50 layers with residual connections
                - Solves vanishing gradient problem
                - Pre-trained on ImageNet
                - Excellent for complex pattern recognition
                """)
            elif selected_model_name == "InceptionV3":
                st.write("""
                **InceptionV3 Architecture:**
                - Multi-scale feature extraction
                - Efficient inception modules
                - Pre-trained on ImageNet
                - Good for diverse image patterns
                """)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# Sidebar with model performance (if available)
st.sidebar.header("📈 Model Performance")
if available_models:
    st.sidebar.write(f"""
    **{selected_model_name} Details:**
    - Input Size: 224×224×3
    - Transfer Learning: ✅
    - Data Augmentation: ✅
    - Dropout: 0.5
    - Optimizer: Adam
    """)
else:
    st.sidebar.write("""
    **Demo Model:**
    - Simple CNN architecture
    - Random weights (not trained)
    - For demonstration only
    """)

st.sidebar.header("📁 File Requirements")
st.sidebar.write("""
**To use trained models:**
1. Place your .h5 model files in the app directory:
   - `parkinsons_vgg16_model.h5`
   - `parkinsons_resnet50_model.h5`
   - `parkinsons_inceptionv3_model.h5`
2. Optionally include `class_indices.json`
3. Restart the app to load models
""")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p>Current Model: {selected_model_name} | Developed for educational purposes | Always consult healthcare professionals for medical advice</p>
</div>
""", unsafe_allow_html=True)