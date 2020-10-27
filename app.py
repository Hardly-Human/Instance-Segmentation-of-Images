import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data, utils

@st.cache(allow_output_mutation=True)
def load_model(model_name):
  model = model_zoo.get_model(model_name, pretrained = True)
  return model

def plot_image(model, x , orig_img):
  st.warning("Inferencing from Model..")
  ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in model(x)]

  width, height = orig_img.shape[1], orig_img.shape[0]
  masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
  orig_img = utils.viz.plot_mask(orig_img, masks)

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                          class_names=model.classes, ax=ax)
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.success("Instance Segmentation Successful!! Plotting Image..")
  st.pyplot(plt.show())

def footer():
  st.markdown("""
  * * *
  Built with ❤️ by [Rehan uddin](https://hardly-human.github.io/)
  """)
  st.success("Rehan uddin (Hardly-Human)👋😉")


################################################################################
# main()
################################################################################

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

	if st.button("Run Model"):
      st.warning("Loading Model..🤞")
      model = load_model('mask_rcnn_resnet50_v1b_coco')
      st.success("Loaded Model Succesfully!!🤩👍")

      x, orig_img = data.transforms.presets.rcnn.load_test(image_path)
      plot_image(model,x,orig_img)



if __name__== "__main__":
  main()
  footer()