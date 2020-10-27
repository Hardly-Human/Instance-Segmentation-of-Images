import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data, utils

def main():
  
  st.title("Instance Segmentation App")
  st.text("Built with gluoncv and Streamlit")
  st.markdown("### [About Instance Segmentation]() `            ` [View Source]()")

  image_file = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])
  
  if image_file is None:
     st.warning("Upload Image and Run Model")

  if image_file is not None:
    image1 = Image.open(image_file)
    rgb_im = image1.convert('RGB') 
    image = rgb_im.save("saved_image.jpg")
    image_path = "saved_image.jpg"

  

if __name__== "__main__":
  main()