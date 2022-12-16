import io
import streamlit as st
import numpy as np
import SZ_plotting as SZplt
from SZ_objects import options_macc, kSZ_halos
# from SZ_CVAE import vae, X_VAL

st.set_page_config(
    page_title="SZ Map Dashboard", 
    page_icon="https://ibb.co/LCn1Wn1",
    layout="wide")
st.title("SZ Map Dashboard")
st.caption("Daniel Farid")
st.caption("AMTH 491, Yale University Program in Applied Mathematics")

SZ_comp, custom_map, cvae = st.tabs(["Data Comparison", "Map Visualizer", "CVAE Map Generation"])

# -------------------- kSZ vs tSZ Map Comparisons -------------------- #

with SZ_comp:
    st.header("kSZ vs. tSZ Map Comparisons")
    st.caption("Compare corresponding kSZ and tSZ maps")

    st.button("Random Halo ID", help="Click to randomize halo ID", on_click=SZplt.randomize_halo_id)
    halo_id = st.number_input('Halo ID', 0, 987, step=1, key='halo_id')

    fig, kSZ, tSZ = SZplt.display_SZ_imgs(halo_id)
    st.pyplot(fig)
    with st.expander("Halo properties"):
        st.latex(f"\\text{{Mass Accretion Rate:}} {kSZ.macc}")
        st.latex(f"\\text{{Mass (m200c)}}: 10^{{{kSZ.m200c}}}")

# -------------------- kSZ Map Visualizer -------------------- #

with custom_map:
    st.header("SZ Map Visualizer")
    st.caption("View tSZ and kSZ maps with specific mass accretion rate and mass")
    macc_choice = st.selectbox("Mass Accretion Rate", options_macc)
    mass_options = SZplt.filter_mass_options(macc_choice)
    if len(mass_options) == 1:
        mass_choice = mass_options[0]
        st.write(f"Mass Accretion Rate (only one option): {mass_choice}")
    else:
        mass_choice = st.select_slider("Mass", mass_options)

    fig, kSZ, tSZ = SZplt.select_img(macc_choice, mass_choice)
    st.pyplot(fig)

    filename = SZplt.save_fig_to_pdf(macc_choice, mass_choice, fig)
    with open(filename, "rb") as img:
        st.download_button(
            "Download Figure", 
            data=img, 
            file_name=f"{filename}.pdf", 
            help=f"Download PDF ({filename}.pdf)"
        )

    with io.BytesIO() as buffer:
        np.savetxt(buffer, kSZ.image, delimiter=",")
        st.download_button(
            "Download kSZ Image Data (CSV)", 
            data=buffer, 
            file_name=f"k{filename}.csv",
            help=f"Download CSV (k{filename}.csv)",
        )

        np.savetxt(buffer, tSZ.image, delimiter=",")
        st.download_button(
            "Download tSZ Image Data (CSV)", 
            data=buffer, 
            file_name=f"t{filename}.csv",
            help=f"Download CSV (t{filename}.csv)"
        )



