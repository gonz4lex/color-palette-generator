import streamlit as st

import palgen

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Color Palette Generator")

with st.sidebar:
    # TODO: sidebar info
    n_colors = st.slider("Number of colors", 3, 8, 4)


## App
# TODO: upload or paste link
uploaded_image = st.file_uploader("Upload an image")

if uploaded_image:
    st.image(uploaded_image) # TODO: optionally display
    image_data = palgen.image_to_df(uploaded_image)
    
    # TODO: cache
    mod = palgen.kmeans(data=image_data, n_colors=n_colors)
    colors = palgen.get_colors(mod=mod)

    # TODO: center and improve chart
    # st.plotly_chart(
    #     palgen.colors_to_image(colors)
    # )
    fig = palgen.colors_to_image(colors)
    st.image(fig.to_image(format="png", width=600, height=350, scale=2))
