import streamlit as st
import numpy as np
import os
from glob import glob
from PIL import Image
from requests_toolbelt import MultipartEncoder
import requests
import io
import base64
from streamlit_option_menu import option_menu

def inference_folder(img_name: str, conf: str):

    m = MultipartEncoder(
        fields={
            'img_name': img_name,
            'conf':conf
        }
    )

    r = requests.post("http://localhost:8000/inference_folder",
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000
                    )

    return r

def inference_input(img, conf: str):

    m = MultipartEncoder(
        fields={
            # 'file': ('filename', image, 'image/jpeg'),
            'img': ('img', img, 'image/jpeg'),
            'conf':conf
        }
    )

    r = requests.post("http://localhost:8000/inference_input",
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000
                    )

    return r

def homepage():
    # Load two images
    st.title("Car Logo Detection with YOLO-NAS")
    image_path1 = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Car_Logo_Dataset_27_Contrasted_COCO_FINAL/test/test_59.jpg'
    image_path2 = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/runs/detect/test_59.jpg'
    image_path3 = '/Users/amade/OneDrive/Desktop/SKRIPSI/gambar_buat_skripsi/hiasan.jpg'
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image3 = Image.open(image_path3)

    # Center the title and images
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div style="margin-right: 20px;">
                <img src="data:image/jpeg;base64,{}" alt="Image 1" style="width: 300px; height: auto;">
                <p>Sample image before car logo detection</p>
            </div>
            <div>
                <img src="data:image/jpeg;base64,{}" alt="Image 2" style="width: 300px; height: auto;">
                <p>Sample image after car logo detection</p>
            </div>
        </div>
    """.format(image_to_base64(image1), image_to_base64(image2)), unsafe_allow_html=True)

    # Apply Markdown styling to the text
    st.markdown("""
        <h2 style="color: #2a4b7c; font-size: 24px;">Welcome to Car Logo Detection App</h2>
        <p style="color: #4c4c4c; font-size: 16px;">
            Bidang deteksi dan identifikasi logo telah membuat kemajuan yang signifikan dalam computer vision dan 
            identifikasi gambar, terutama di arena aplikasi otomotif. Deteksi dan klasifikasi logo terutama yang terkait 
            dengan merek mobil memiliki pengaruh yang besar dalam berbagai konteks dunia nyata, dari sistem transportasi 
            cerdas hingga kontrol lalu lintas.
        </p>
        <strong style="color: #006633;">Get started:</strong> Navigate to the Detection Page.
        
        <em style="color: #990000;">Let's explore the world of car logos!</em>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <img src="data:image/jpeg;base64,{}" alt="Image 2" style="width: 300px; height: auto;">
                <p>Sample image after car logo detection</p>
            </div>
        </div>
    """.format(image_to_base64(image3)), unsafe_allow_html=True)

    st.markdown("""
        <h2 style="color: #2a4b7c; font-size: 24px;">A Little About YOLO-NAS</h2>
        <p style="color: #4c4c4c; font-size: 16px;">
            Deci, sebuah perusahaan yang menciptakan model dan alat-alat tingkat produksi untuk membangun, 
            mengoptimalkan, dan mengimplementasikan model pembelajaran mendalam, menerbitkan YOLO-NAS pada Mei 2023. 
            YOLO-NAS cocok untuk aplikasi edge-device real-time karena kemampuannya untuk mengidentifikasi objek kecil,
            meningkatkan lokalisasi dan presisi, dan meningkatkan perbandingan kinerja per komputasi. 
            Arsitektur open-source juga dapat digunakan dalam penelitian.
        </p>
        <strong style="color: #4c4c4c; font-size: 16px;">
            •	Quantization Aware Blocks and Selective Quantization
        </strong>
        <p style="color: #4c4c4c; font-size: 16px;">
            YOLO-NAS secara unik mengintegrasikan quantization aware blocks dan mengadopsi strategi kuantisasi hibrida, yang memastikan kinerja yang unggul, terutama di lingkungan dengan sumber terbatas.
        </p>
        <strong style="color: #4c4c4c; font-size: 16px;">
            •	Quantization Aware Blocks and Selective Quantization
        </strong>
        <p style="color: #4c4c4c; font-size: 16px;">
            YOLO-NAS secara unik mengintegrasikan quantization aware blocks dan mengadopsi strategi kuantisasi hibrida, yang memastikan kinerja yang unggul, terutama di lingkungan dengan sumber terbatas.
        </p>
        <strong style="color: #4c4c4c; font-size: 16px;">
            •	Detection Head with Distribution Probability for Size Regression
        </strong>
        <p style="color: #4c4c4c; font-size: 16px;">
            Desain detection head ini sangat berguna dalam scenario Dimana ukuran objek yang akan dideteksi memiliki ukuran yang bervariasi secara signifikan. Tidak hanya satu ukuran tetap yang diprediksi, melainkan berbagai ukuran yang mungkin, YOLO-NAS meningkatkan akurasi dalam mendeteksi objek dengan skala yang bervariasi. 
        </p>
        <strong style="color: #4c4c4c; font-size: 16px;">
            •	NAS-Generated Backbone Design with AutoNAC
        </strong>
        <p style="color: #4c4c4c; font-size: 16px;">
            Pendekatan ini memungkinkan YOLO-NAS untuk memiliki struktur jaringan yang memenuhi persyaratan spesifik tugas deteksi objek dengan akurasi yang tidak tertandingi. AutoNAC memainkan peran penting dalam menentukan ukuran dan struktur optimal dalam backbone YOLO-NAS, termasuk mengkonfigurasi jenis blok, jumlah blok, dan jumlah saluran di setiap tahap.
        </p>
    """, unsafe_allow_html=True)


    # Additional styling options
    st.markdown("""
        <style>
            body {
                color: #444;
                background-color: #f8f9fa;
            }
            .sidebar .sidebar-content {
                background-color: #343a40;
                color: #ffffff;
            }
            .streamlit-container {
                max-width: 1200px;
            }
        </style>
    """, unsafe_allow_html=True)

# Helper function to convert image to base64
def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Detection Page (Your existing code)
def detection_page():
    st.title("Get Started on Car Logo Detection")
    # Sidebar styling
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
    # Custom styles for text in the sidebar
    sidebar_text_styles = {
        'font-size': '20px',  # Adjust font size for the sidebar text
        'color': 'green'  ,
        'margin-bottom': '0',    # Adjust font color for the sidebar text
    }
    sidebar_text_styles_2 = {
        'font-size': '20px',  # Adjust font size for the sidebar text
        'color': 'red'  ,
        'margin-bottom': '0',    # Adjust font color for the sidebar text
    }

    # Apply styles to 'Choose the app mode'
    st.sidebar.markdown(f'<p style="{";".join([f"{key}: {value}" for key, value in sidebar_text_styles.items()])}">Choose the app mode</p>', unsafe_allow_html=True)
    app_model = st.sidebar.selectbox('', ('Select from Folder', 'Input Image'))

    if app_model == 'Select from Folder':
        st.sidebar.markdown(f'<p style="{";".join([f"{key}: {value}" for key, value in sidebar_text_styles_2.items()])}">Challenges</p>', unsafe_allow_html=True)

        chosen_folder = st.sidebar.selectbox('', ('Blur Images', 'Very Small Logos', 'Multiple Logos Detected', 'Difficult Angles', 'Normal'))
        
        if chosen_folder == 'Blur Images':
            base_folder_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Streamlit_Image_Folder/Blur/'
            
            files = os.listdir(base_folder_path)
            selected_image = st.sidebar.selectbox("Select an image for inference", files)

            # Display the selected image
            selected_image_path = os.path.join(base_folder_path, selected_image)
            selected_image_display = Image.open(selected_image_path)
            st.sidebar.image(selected_image_display, caption="Selected Image", use_column_width=True)

            if st.sidebar.button("Perform Inference"):
                result = inference_folder(selected_image, str(confidence))
                data_out = result.json()
                img = Image.open(io.BytesIO(base64.b64decode(data_out['inf_img'])))

                # Display the image with bounding boxes
                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

                # Display detected labels
                st.markdown(f'<strong class="st-cu">No logo detected because of blurred images of a car is usually a common case especially images from dashcams or traffic cams, YOLO NAS can perform detection well at a certain level of blurring</strong>', unsafe_allow_html=True)
        elif chosen_folder == 'Very Small Logos':
            base_folder_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Streamlit_Image_Folder/Very_Small_Logos/'
            
            files = os.listdir(base_folder_path)
            selected_image = st.sidebar.selectbox("Select an image for inference", files)

            # Display the selected image
            selected_image_path = os.path.join(base_folder_path, selected_image)
            selected_image_display = Image.open(selected_image_path)
            st.sidebar.image(selected_image_display, caption="Selected Image", use_column_width=True)

            if st.sidebar.button("Perform Inference"):
                result = inference_folder(selected_image, str(confidence))
                data_out = result.json()
                img = Image.open(io.BytesIO(base64.b64decode(data_out['inf_img'])))

                # Display the image with bounding boxes
                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

                # Display detected labels
                st.markdown(f'<strong class="st-cu">YOLO NAS can detect small object better due to the architecture of the model and its DFL technique, which is very compatible for detecting car logos</strong>', unsafe_allow_html=True)
        elif chosen_folder == 'Multiple Logos Detected':
            base_folder_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Streamlit_Image_Folder/Multiple_Detections/'
            
            files = os.listdir(base_folder_path)
            selected_image = st.sidebar.selectbox("Select an image for inference", files)

            # Display the selected image
            selected_image_path = os.path.join(base_folder_path, selected_image)
            selected_image_display = Image.open(selected_image_path)
            st.sidebar.image(selected_image_display, caption="Selected Image", use_column_width=True)

            if st.sidebar.button("Perform Inference"):
                result = inference_folder(selected_image, str(confidence))
                data_out = result.json()
                img = Image.open(io.BytesIO(base64.b64decode(data_out['inf_img'])))

                # Display the image with bounding boxes
                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

                # Display detected labels
                st.markdown(f'<strong class="st-cu">YOLO NAS can detect multiple logos in one image better than DETR or YOLOv7</strong>', unsafe_allow_html=True)
        elif chosen_folder == 'Difficult Angles':
            base_folder_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Streamlit_Image_Folder/Difficult_Angles/'
            
            files = os.listdir(base_folder_path)
            selected_image = st.sidebar.selectbox("Select an image for inference", files)

            # Display the selected image
            selected_image_path = os.path.join(base_folder_path, selected_image)
            selected_image_display = Image.open(selected_image_path)
            st.sidebar.image(selected_image_display, caption="Selected Image", use_column_width=True)

            if st.sidebar.button("Perform Inference"):
                result = inference_folder(selected_image, str(confidence))
                data_out = result.json()
                img = Image.open(io.BytesIO(base64.b64decode(data_out['inf_img'])))

                # Display the image with bounding boxes
                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

                # Display detected labels
                st.markdown(f'<strong class="st-cu">YOLO NAS can detect car logos with a difficult angle better than YOLOv7, 
                            even though not all of the detected logos are very accurate</strong>', unsafe_allow_html=True)
        elif chosen_folder == 'Normal':
            base_folder_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Streamlit_Image_Folder/Normal/'
            
            files = os.listdir(base_folder_path)
            # Display the selected image
            selected_image_path = os.path.join(base_folder_path, selected_image)
            selected_image_display = Image.open(selected_image_path)
            st.sidebar.image(selected_image_display, caption="Selected Image", use_column_width=True)

            selected_image_path = os.path.join(base_folder_path, selected_image)

            if st.sidebar.button("Perform Inference"):
                result = inference_folder(selected_image, str(confidence))
                data_out = result.json()
                img = Image.open(io.BytesIO(base64.b64decode(data_out['inf_img'])))

                # Display the image with bounding boxes
                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

                # Display detected labels
                st.markdown(f'<p class="st-cu">This is just an example of a normal image which a YOLO-NAS model can perform car logo detection well</p>', unsafe_allow_html=True)

    elif app_model == 'Input Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        # Display the selected image
        if uploaded_file:
            selected_image_display = Image.open(uploaded_file)
            st.sidebar.image(selected_image_display, caption="Selected Image", use_column_width=True)

        if st.button("Perform Inference"):
            if uploaded_file:
                result = inference_input(uploaded_file, str(confidence))
                data_out = result.json()
                img = Image.open(io.BytesIO(base64.b64decode(data_out['inf_img'])))

                # Display the image with bounding boxes
                st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)
    
    st.sidebar.title("How to Use")
    st.sidebar.markdown("""
    To do car logo detection:

    1. Click the 'predict' button in the Navigation section in the sidebar
    2. Choose the app mode you want to use
    - If you choose 'Input Image' mode, please select a image by clicking on the Browse Images button
    - If you choose 'Select from Folder' mode, please continue by choosing the challenges in the dropdown, then choose the image from the folder
    3. After choosing the image, click on the Inference button to do car logo detection
    """)

def contact_page():
    st.markdown("""
        <h2 style="color: #2a4b7c; font-size: 24px;">If you have any questions or feedback, feel free to contact us:</h2>
        <strong style="color: #006633;">This website is made by Yohanes Amadeo Marvell (2301862260) for thesis purposes</strong>
        <strong style="color: #006633;">Email:</strong> yohanes.marvell@binus.ac.id
        <strong style="color: #990000;">Phone:</strong> 0812-3456-789
    """, unsafe_allow_html=True)


selected = option_menu(
    menu_title = None,
    options = ["Home", "Detection", "Contact"],
    icons = ["house", "search", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link":{"--hover-color": "lightgray"}
    }
)

if selected == "Home":
    homepage()
elif selected == "Detection":
    detection_page()
elif selected == "Contact":
    contact_page()




