import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import streamlit as st
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from SZ_objects import tSZ_halos, kSZ_halos

def display_imgs(kSZ, tSZ):
    fig, ax = plt.subplots(1,2, figsize=(5,5))

    # tSZ
    im0 = ax[0].imshow(tSZ.image, vmin=0, vmax=2e-6)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im0, cax=cax, orientation='vertical')
    cbar.ax.tick_params(size=2, labelsize=5)
    cbar.ax.yaxis.get_offset_text().set_fontsize(5)
    ax[0].text(1, 10, 'tSZ', fontsize=10, color="white")
    
    # kSZ
    im1 = ax[1].imshow(kSZ.image, vmin=-2e-7, vmax=2e-7)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar.ax.tick_params(size=2, labelsize=5)
    cbar.ax.yaxis.get_offset_text().set_fontsize(5)
    ax[1].text(1, 10, 'kSZ', fontsize=10)

    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    
    return fig, kSZ, tSZ

def display_SZ_imgs(halo_id: int):
    kSZ = kSZ_halos[halo_id]
    tSZ = tSZ_halos[halo_id]
    return display_imgs(kSZ, tSZ)

def randomize_halo_id():
    st.session_state.halo_id = np.random.choice(987)

def select_img(macc, m200c):
    tSZ = next((obj for obj in tSZ_halos if floor(obj.macc) == macc and obj.m200c == m200c), None)
    kSZ = next((obj for obj in kSZ_halos if floor(obj.macc) == macc and obj.m200c == m200c), None)
    return display_imgs(kSZ, tSZ)

@st.cache
def filter_mass_options(macc: int):
    mass_options = np.unique([halo.m200c for halo in kSZ_halos if floor(halo.macc) == macc])
    mass_options.sort()
    return mass_options

def save_fig_to_pdf(macc: int, mass: float, fig: matplotlib.figure.Figure):
    filename = f"SZ_{macc}_{np.round(mass, 3)}"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    return filename
