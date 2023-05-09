import pandas as pd
import numpy as np

import plotly.express as px

from PIL import Image
import colorsys

from sklearn.cluster import KMeans

def image_to_df(image_path):
    # Open the image and convert it to RGB if it is in RGBA format
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        # Convert the image to values and pass into datafarme
        rgb_values = list(image.getdata())
    df = pd.DataFrame(rgb_values, columns=["r", "g", "b"])
    return df


def kmeans(data, n_colors: int=6):
    mod = KMeans(n_clusters=n_colors, init="k-means++", n_init="auto").fit(data)
    return mod

def get_colors(mod):
    centers = mod.cluster_centers_.astype(int)
    return centers.tolist()

def get_hsv_value(rgb):
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r, g, b)[2]

def sort(colors):
    return colors.sort(key=get_hsv_value, reverse=True)

def colors_to_image(colors):
    img_rgb = np.array([colors], dtype=np.uint8)

    fig = px.imshow(img_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

if __name__ == "__main__":
    data = image_to_df("./img/mountains.jpg")

    kmeans_mod = kmeans(data=data)

    colors = get_colors(mod=kmeans_mod)
    sorted_colors = sort(colors=colors)

    colors_img = colors_to_image(colors=colors)
    


