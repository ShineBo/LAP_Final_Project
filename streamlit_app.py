import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Flower Classification App",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 10px 0;
    }
    .confidence-high {
        color: #228B22;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF8C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #DC143C;
        font-weight: bold;
    }
    .error-box {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model = tf.keras.models.load_model('flower_classifier_final.h5')
        return model
    except:
        st.error("❌ Model file not found! Please make sure 'flower_classifier_final.h5' exists in the same directory.")
        return None

def preprocess_image(image):
    """Preprocess the image for prediction with proper channel handling"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA image
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 1:  # Single channel
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Ensure we have 3 channels (RGB)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError("Unable to convert image to RGB format")
        
        # Resize image to model input size
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def is_likely_flower(predictions, threshold=0.3):
    """Check if the image is likely to contain a flower based on prediction confidence"""
    max_confidence = max(predictions.values())
    return max_confidence >= threshold

def predict_flower(model, image):
    """Make prediction on the image with comprehensive error handling"""
    # Class names
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get all predictions for visualization
        prediction_dict = {}
        for i, class_name in enumerate(class_names):
            prediction_dict[class_name] = predictions[0][i]
        
        # Check if this looks like a flower
        if not is_likely_flower(prediction_dict, threshold=0.2):
            return None, 0, prediction_dict, "The system could not detect a flower in this image. Please try uploading a clear image of a flower."
        
        return class_names[predicted_class_idx], confidence, prediction_dict, None
        
    except Exception as e:
        error_msg = "The system encountered an issue processing your image. Please try with a different image."
        return None, 0, {}, error_msg

def create_confidence_chart(predictions):
    """Create a confidence chart using Plotly"""
    flowers = list(predictions.keys())
    confidences = [predictions[flower] * 100 for flower in flowers]
    
    # Create discrete colors based on confidence levels
    colors = []
    for conf in confidences:
        if conf < 20:
            colors.append('#FF6B6B')  # Red for low confidence
        elif conf < 50:
            colors.append('#FFE66D')  # Yellow for medium confidence  
        else:
            colors.append('#4ECDC4')  # Green for high confidence
    
    # Sort data by confidence (highest to lowest)
    sorted_data = sorted(zip(flowers, confidences, colors), key=lambda x: x[1], reverse=True)
    sorted_flowers, sorted_confidences, sorted_colors = zip(*sorted_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(sorted_flowers),
            y=list(sorted_confidences),
            marker_color=list(sorted_colors),
            text=[f'{conf:.1f}%' for conf in sorted_confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence for Each Flower Type",
        xaxis_title="Flower Type",
        yaxis_title="Confidence (%)",
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=400,
        showlegend=False
    )
    
    return fig

def get_flower_info(flower_name):
    """Get information about the predicted flower"""
    flower_info = {
        'dandelion': {
            'emoji': '🌼',
            'description': 'Dandelions are bright yellow flowers that are often considered weeds but are actually edible and medicinal.',
            'facts': ['Can be eaten raw or cooked', 'Rich in vitamins A, C, and K', 'Blooms almost year-round']
        },
        'daisy': {
            'emoji': '🌻',
            'description': 'Daisies are simple, cheerful flowers with white petals and yellow centers, symbolizing innocence and purity.',
            'facts': ['Symbol of new beginnings', 'Belongs to the Asteraceae family', 'Can bloom from spring to fall']
        },
        'rose': {
            'emoji': '🌹',
            'description': 'Roses are classic flowers known for their beauty and fragrance, often associated with love and romance.',
            'facts': ['Over 150 species exist', 'Can live for decades with proper care', 'Petals are often used in perfumes']
        },
        'sunflower': {
            'emoji': '🌻',
            'description': 'Sunflowers are large, bright yellow flowers that turn to follow the sun throughout the day.',
            'facts': ['Can grow up to 12 feet tall', 'Seeds are rich in healthy fats', 'Native to North America']
        },
        'tulip': {
            'emoji': '🌷',
            'description': 'Tulips are elegant spring flowers that come in many colors and are symbols of perfect love.',
            'facts': ['Originally from Central Asia', 'Bloom for only 3-7 days', 'Netherlands is famous for tulip cultivation']
        }
    }
    
    return flower_info.get(flower_name, {
        'emoji': '🌸',
        'description': 'A beautiful flower!',
        'facts': ['Nature is amazing!']
    })

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Flower Classification App 🌸</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About This App")
        st.write("""
        This app uses deep learning to classify flower images into 5 categories:
        - 🌼 Daisy
        - 🌻 Dandelion  
        - 🌹 Rose
        - 🌻 Sunflower
        - 🌷 Tulip
        """)
        
        st.header("📊 Model Info")
        st.write("""
        - **Architecture**: MobileNetV2 + Custom layers
        - **Input Size**: 224x224 pixels
        - **Training**: Transfer learning with fine-tuning
        """)
        
        st.header("🔧 How to Use")
        st.write("""
        1. Upload an image of a flower
        2. Wait for the model to process
        3. View the prediction and confidence
        4. Explore the detailed analysis
        """)
        
        st.header("💡 Tips for Best Results")
        st.write("""
        - Use clear, well-lit images
        - Ensure the flower is the main subject
        - Avoid blurry or low-quality images
        - JPG and PNG formats work best
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Your Flower Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of a flower for best results"
        )
        
        # Use uploaded file
        image_source = uploaded_file
        
        if image_source is not None:
            # Display the image
            image = Image.open(image_source)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add image info
            st.write(f"**Image size**: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Image mode**: {image.mode}")
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if image_source is not None:
            with st.spinner("🤖 Analyzing the flower..."):
                # Make prediction
                predicted_class, confidence, all_predictions, error_message = predict_flower(model, image)
                
                if error_message:
                    # Display error message in a friendly way
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>🚫 Unable to Classify</h3>
                        <p>{error_message}</p>
                        <p><strong>Suggestions:</strong></p>
                        <ul>
                            <li>Make sure the image contains a clear flower</li>
                            <li>Try a different angle or lighting</li>
                            <li>Ensure the flower is the main subject</li>
                            <li>Use a higher quality image</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if all_predictions:  # Show predictions even if confidence is low
                        st.subheader("📊 Raw Predictions (Low Confidence)")
                        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                        for flower, conf in sorted_predictions:
                            st.write(f"• {flower.title()}: {conf:.2%}")
                        
                        st.info("💡 These predictions have very low confidence, suggesting the image may not contain a recognizable flower.")
                
                else:
                    # Display successful prediction
                    flower_info = get_flower_info(predicted_class)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>{flower_info['emoji']} Predicted Flower: {predicted_class.title()}</h2>
                        <h3 class="{'confidence-high' if confidence > 0.8 else 'confidence-medium' if confidence > 0.5 else 'confidence-low'}">
                            Confidence: {confidence:.2%}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence interpretation
                    if confidence > 0.8:
                        st.success("🎯 High confidence prediction!")
                    elif confidence > 0.5:
                        st.warning("⚠️ Medium confidence - consider a clearer image")
                    else:
                        st.info("ℹ️ Low confidence - the model is uncertain about this prediction")
                    
                    # Flower information
                    st.subheader("📖 About This Flower")
                    st.write(flower_info['description'])
                    
                    st.subheader("💡 Fun Facts")
                    for fact in flower_info['facts']:
                        st.write(f"• {fact}")
        else:
            st.info("👆 Please upload an image or take a photo to start classification")
    
    # Detailed analysis section
    if image_source is not None and 'all_predictions' in locals() and all_predictions:
        st.header("📊 Detailed Analysis")
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # Confidence chart
            fig = create_confidence_chart(all_predictions)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("🏆 All Predictions")
            
            # Sort predictions by confidence
            sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            
            for i, (flower, conf) in enumerate(sorted_predictions):
                flower_emoji = get_flower_info(flower)['emoji']
                
                if i == 0:  # Top prediction
                    st.markdown(f"**🥇 {flower_emoji} {flower.title()}**: {conf:.2%}")
                elif i == 1:  # Second prediction
                    st.markdown(f"🥈 {flower_emoji} {flower.title()}: {conf:.2%}")
                elif i == 2:  # Third prediction
                    st.markdown(f"🥉 {flower_emoji} {flower.title()}: {conf:.2%}")
                else:
                    st.markdown(f"{flower_emoji} {flower.title()}: {conf:.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Streamlit and TensorFlow<br>
        Upload clear, well-lit flower images for best results!
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()