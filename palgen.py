import pandas as pd
import numpy as np

import plotly.express as px

from PIL import Image
import colorsys

from sklearn.cluster import KMeans

def img_to_df(image_path):
    # Open the image and convert it to RGB if it is in RGBA format
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        # Convert the image to values and pass into datafarme
        rgb_values = list(img.getdata())
    df = pd.DataFrame(rgb_values, columns=["r", "g", "b"])
    return df


def kmeans(data):
    mod = KMeans(n_clusters=6, init="k-means++", n_init="auto").fit(data)
    return mod

def get_palette(mod):
    centers = mod.cluster_centers_.astype(int)
    return centers.tolist()

def get_hsv_value(rgb):
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r, g, b)[2]

def sort(palette):
    return palette.sort(key=get_hsv_value, reverse=True)

def palette_to_img(palette):
    img_rgb = np.array([palette], dtype=np.uint8)

    fig = px.imshow(img_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

    return fig

if __name__ == "__main__":
    data = img_to_df("./img/mountains.jpg")

    kmeans_mod = kmeans(data=data)

    palette = get_palette(mod=kmeans_mod)
    # TODO:
    #  sorted_palette = sort(palette=palette)

    palette_img = palette_to_img(palette=palette)
    
    # TODO:
    # palette_img.save("./palette.png")


