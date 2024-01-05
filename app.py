import streamlit as st
from detect_images_streamlit import *
from PIL import Image

def main():
    st.title("Car Logo Detection with YOLO-NAS")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
        width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
        width: 300px;
        margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose an image...", type= ["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Perform inference on the uploaded image
        img, labels = perform_inference(uploaded_file)

        # Display the image with bounding boxes
        st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

        # Display detected labels
        st.write(f"Detected labels: {labels}")
    st.sidebar.text('Original Image')
    # st.sidebar.image(uploaded_file)
    perform_inference(uploaded_file)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

