
import streamlit as st
from SZ_CVAE import generate_map
from Map_Visualization import download_fig
import matplotlib.pyplot as plt
import numpy as np

def plot_generated_map(macc: float, mass: float):   
    image_data = generate_map(macc, mass)
    plt.imshow(image_data)
    plt.colorbar()
    plt.yticks([])
    plt.xticks([])
    plt.text(1, 8, "kSZ", fontsize=20)
    plt.text(95, 8, f"{np.round(macc, 2)}, {np.round(mass, 2)}", fontsize=15)
    st.pyplot(plt)

st.set_page_config(page_title="CVAE Map Generation (Beta)")
st.header("Generate kSZ Images with CosmicVAE")
st.write("Generate novel kSZ images with a conditional variational autoencoder (CVAE).")
st.caption("This model is in its preliminary results phase. It shows promise of being able to generate complex kSZ maps given fine tuning.")
mass_choice = st.number_input("Mass", value=14.0, help="Input desired mass of generated kSZ image")
macc_choice = st.number_input("Mass Accretion Rate", value=3.0, help="Input desired mass accretion rate of generated kSZ image")
fig = st.button("Generate Image", help="Generate kSZ Image with CosmicVAE", on_click=plot_generated_map, args=(macc_choice, mass_choice))
