import streamlit as st
import cv2
import pandas as pd
import yaml

from d_infer_load_render import infer_load_render

st.set_page_config(
    page_title="BuildingDAG",
    page_icon="🏘️",
    layout="centered",
)

# st.title("BuildingDAG")
sketch_img_path = "./inference/sketch.png"
output_img_path = "./inference/output.png"
output_yml_path = "./inference/output.yml"

upload_img = st.file_uploader("Upload Image (building sketch)", type=["jpg", "jpeg", "png"])
if upload_img:
    with open(sketch_img_path, "wb") as f:
        f.write(upload_img.read())
    img = cv2.imread(sketch_img_path)
    st.image(sketch_img_path, caption="Input", use_column_width=True)

if st.button("Inference & Load Param & Render"):
    infer_load_render()
    img = cv2.imread(output_img_path)
    st.image(img, caption="Output", use_column_width=True)
    with open(output_yml_path, "r") as f:
        df = yaml.load(f, Loader=yaml.FullLoader)
    for (key, val) in df.items():
        if type(val) == list:
            df[key] = str(val)  # make sure pd doesn't pad other entries into a list
    df = pd.DataFrame(df, index=[0])
    st.dataframe(df, hide_index=True)
    st.caption("Predicted Shape Parameters↑")
