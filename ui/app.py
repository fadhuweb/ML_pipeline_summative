import streamlit as st
import requests
import time
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    layout="wide",
    page_icon="🏥"
)

# Custom CSS for better styling including metric readability fix
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Metric color fix added here */
    .stMetric label, .stMetric div {
        color: black !important;
    }

    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 20px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Chest X-ray Pneumonia ML Pipeline Dashboard")

API_BASE_URL = "http://localhost:8000"
TRAIN_DIR = r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\train"

# Initialize session state
if 'metrics_cache' not in st.session_state:
    st.session_state.metrics_cache = {}
if 'last_selected_model' not in st.session_state:
    st.session_state.last_selected_model = None

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🔍 Predict", "⚙️ Training", "📊 Metrics", "📈 Visualizations"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Fetch available models
try:
    models_response = requests.get(f"{API_BASE_URL}/models").json()
    available_models = models_response.get("valid_model_files", [])
    selected_model = models_response.get("currently_selected", None)
except Exception:
    available_models = []
    selected_model = None

st.sidebar.subheader("🤖 Model Selection")
if available_models:
    selected_model = st.sidebar.selectbox(
        "Choose model",
        options=available_models,
        index=available_models.index(selected_model) if selected_model in available_models else 0
    )
    if st.sidebar.button("Apply Selection", type="primary"):
        try:
            r = requests.post(f"{API_BASE_URL}/select-model", params={"model_name": selected_model})
            if r.status_code == 200:
                st.sidebar.success(f"Model {selected_model} selected")
                if st.session_state.last_selected_model != selected_model:
                    st.session_state.metrics_cache = {}
                    st.session_state.last_selected_model = selected_model
                st.rerun()
            else:
                st.sidebar.error("Selection failed")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
else:
    st.sidebar.info("No models available. Upload or retrain a model first.")

st.sidebar.markdown("---")

# Page 1 predict
if page == "🔍 Predict":
    st.header("Single Image Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"]
        )
        
        if file:
            from PIL import Image
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write(f"Filename: {file.name}")
            st.write(f"Size: {file.size / 1024:.2f} KB")
    
    with col2:
        st.subheader("Prediction Result")
        
        if file:
            if st.button("Run Prediction", type="primary", use_container_width=True):
                with st.spinner("Analyzing image"):
                    try:
                        file.seek(0)
                        
                        files = {"file": (file.name, file, file.type)}
                        r = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
                        if r.status_code == 200:
                            result = r.json()
                            
                            prediction = result.get('label', 'Unknown')
                            confidence = result.get('confidence', 0)
                            
                            if prediction.lower() == "pneumonia":
                                st.error("Pneumonia detected")
                            else:
                                st.success("Normal")
                            
                            st.metric("Confidence", f"{confidence * 100:.2f} percent")
                            st.progress(confidence)
                            
                            with st.expander("Raw Result"):
                                st.json(result)
                        else:
                            st.error(f"Prediction failed: {r.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Upload an image to start prediction")

# Page 2 training
elif page == "⚙️ Training":
    st.header("Training and Retraining")
    
    tab1, tab2 = st.tabs(["Upload Data", "Retrain Model"])
    
    with tab1:
        st.subheader("Upload Training Data")
        
        files = st.file_uploader(
            "Choose files",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True
        )

        if files:
            st.success(f"{len(files)} file or files ready for upload")
            
            with st.expander("View Files"):
                for f in files:
                    st.write(f"{f.name} {f.size / 1024:.2f} KB")
            
            if st.button("Upload to Server", type="primary"):
                with st.spinner("Uploading files"):
                    try:
                        file_payload = [("files", (f.name, f, f.type)) for f in files]
                        r = requests.post(f"{API_BASE_URL}/upload", files=file_payload, timeout=120)
                        if r.status_code == 200:
                            st.success("Files uploaded successfully")
                            st.json(r.json())
                        else:
                            st.error(f"Upload failed: {r.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Configure Training Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5)
        
        with col2:
            batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
        
        with col3:
            fine_tune = st.checkbox("Fine tune", value=True)
        
        st.markdown("---")
        
        if st.button("Start Training", type="primary", use_container_width=True):
            st.warning("Training will start. Do not close this page.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Starting training")
            progress_bar.progress(10)
            
            try:
                r = requests.post(
                    f"{API_BASE_URL}/retrain",
                    params={"epochs": epochs, "batch_size": batch_size, "fine_tune": fine_tune},
                    timeout=1800
                )
                
                progress_bar.progress(100)
                status_text.text("Training completed")
                
                if r.status_code == 200:
                    result = r.json()
                    st.success("Training completed successfully")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("Training Summary")
                        if "summary" in result:
                            for key, value in result["summary"].items():
                                if isinstance(value, float):
                                    st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                    
                    with col2:
                        st.markdown("New Model")
                        st.code(result.get("model_path", "Not found"))
                    
                    with st.expander("Full Results"):
                        st.json(result)
                    
                    st.session_state.metrics_cache = {}
                else:
                    st.error(f"Training failed: {r.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page 3 metrics
elif page == "📊 Metrics":
    st.header("Model Performance Metrics")
    
    cache_key = selected_model if selected_model else "latest"
    
    if cache_key in st.session_state.metrics_cache:
        st.info("Using cached metrics. Refresh to recalculate.")
        metrics = st.session_state.metrics_cache[cache_key]
        show_metrics = True
    else:
        metrics = None
        show_metrics = False
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not show_metrics:
            st.warning("No cached metrics available.")
    
    with col2:
        if st.button("Refresh Metrics", type="primary", use_container_width=True):
            with st.spinner("Evaluating metrics"):
                try:
                    r = requests.get(f"{API_BASE_URL}/metrics", timeout=1200)
                    if r.status_code == 200:
                        metrics = r.json()
                        st.session_state.metrics_cache[cache_key] = metrics
                        show_metrics = True
                        st.success("Metrics updated")
                        st.rerun()
                    else:
                        st.error(f"Failed: {r.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if show_metrics and metrics:
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f} percent")
        
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f} percent")
        
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f} percent")
        
        with col4:
            st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
        
        with col5:
            st.metric("Loss", f"{metrics.get('loss', 0):.4f}")
        
        st.markdown("Performance Visualization")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'AUC']
        metric_values = [
            metrics.get('accuracy', 0) * 100,
            metrics.get('precision', 0) * 100,
            metrics.get('recall', 0) * 100,
            metrics.get('auc', 0) * 100
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(metric_names, metric_values, color=colors)
        
        ax.set_ylabel('Score percent', fontsize=12)
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{height:.1f} percent',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        
        st.caption(f"Metrics calculated for model: {cache_key}")

# Page 4 visualizations
elif page == "📈 Visualizations":
    st.header("Dataset Visualizations")
    
    @st.cache_data(show_spinner=False)
    def load_image_data(train_dir):
        classes = []
        class_counts = {}
        image_paths = []
        
        for item in os.listdir(train_dir):
            item_path = os.path.join(train_dir, item)
            if os.path.isdir(item_path):
                if item.lower() == "new_uploads":
                    continue
                
                classes.append(item)
                imgs = [os.path.join(item_path, f) for f in os.listdir(item_path) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                image_paths.extend(imgs)
                class_counts[item] = len(imgs)
        
        return classes, class_counts, image_paths
    
    try:
        classes, class_counts, image_paths = load_image_data(TRAIN_DIR)
        
        st.subheader("Class Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            colors = ['#2ecc71' if 'NORMAL' in cls.upper() else '#e74c3c' for cls in classes]
            bars = ax1.bar(classes, [class_counts[cls] for cls in classes], color=colors)
            ax1.set_ylabel('Number of Images', fontsize=12)
            ax1.set_xlabel('Class', fontsize=12)
            ax1.set_title('Sample Count by Class', fontsize=14, fontweight='bold')
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.pie([class_counts[cls] for cls in classes], 
                    labels=classes,
                    autopct='%1.1f percent',
                    colors=colors,
                    startangle=90)
            ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig2)
        
        st.write("Shows the balance between classes in the dataset.")
        
        total = sum(class_counts.values())
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Images", f"{total:,}")
        with col2:
            normal = class_counts.get('NORMAL', 0)
            st.metric("Normal", f"{normal:,}")
        with col3:
            pneumonia = class_counts.get('PNEUMONIA', 0)
            st.metric("Pneumonia", f"{pneumonia:,}")
        
        st.markdown("---")
        
        st.subheader("Image Brightness Analysis")
        
        with st.spinner("Analyzing brightness"):
            brightness_data = []
            
            for cls in classes:
                cls_path = os.path.join(TRAIN_DIR, cls)
                imgs = [f for f in os.listdir(cls_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))][:100]
                
                for img_file in imgs:
                    try:
                        img = cv2.imread(os.path.join(cls_path, img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (64, 64))
                            brightness_data.append({
                                'Class': cls,
                                'Brightness': np.mean(img)
                            })
                    except:
                        continue
            
            if brightness_data:
                df_brightness = pd.DataFrame(brightness_data)
                
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                
                for cls in classes:
                    cls_data = df_brightness[df_brightness['Class'] == cls]['Brightness']
                    color = '#2ecc71' if 'NORMAL' in cls.upper() else '#e74c3c'
                    ax3.hist(cls_data, bins=30, alpha=0.6, label=cls, color=color)
                
                ax3.set_xlabel('Average Brightness', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                ax3.set_title('Brightness Distribution by Class', fontsize=14, fontweight='bold')
                ax3.legend()
                
                st.pyplot(fig3)
                st.write("Brightness differences between normal and pneumonia X rays.")
        
        st.markdown("---")
        
        st.subheader("Sample Images")
        
        selected_class = st.selectbox("Select Class to View", classes)
        
        class_path = os.path.join(TRAIN_DIR, selected_class)
        sample_images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))][:6]
        
        cols = st.columns(3)
        for idx, img_file in enumerate(sample_images):
            with cols[idx % 3]:
                from PIL import Image
                img = Image.open(os.path.join(class_path, img_file))
                st.image(img, caption=img_file, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")
