from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import pandas as pd
from math import floor
import streamlit as st
import plotly.express as px
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

@dataclass
class SZ:
    """
    SZ object for a single halo, gets sorted by mass accretion rate (macc),
    and can be searched by mass accretion rate value
    
    Args:
        image (np.ndarray): data to show heatmap image
        macc (np.float64): mass accretion rate
        m200c (np.float64): halo mass
    """
    halo_id: int
    image: np.ndarray
    macc: np.float64
    m200c: np.float64
    
    def __lt__(self, other):
        return self.macc < other.macc
    
    def __eq__(self, other):
        return self.macc == other

class tSZ(SZ):
    """
    SZ object specifically for tSZ
    """

class kSZ(SZ):
    """
    SZ object specifically for kSZ
    """

tSZ_maps = np.load('tSZ_128.npy')
kSZ_maps = np.load('kSZ_128.npy')

macc = np.load('macc_200c.npy')
m200c = np.load('m200c.npy')

macc[macc > 10] = 10
macc = np.repeat(macc, 3)
m200c = np.repeat(m200c, 3)

tSZ_halos = [tSZ(halo_id, image, macc, m200c) for halo_id, (image, macc, m200c) in enumerate(zip(tSZ_maps, macc, m200c))]
tSZ_halos.sort()

kSZ_halos = [kSZ(halo_id, image, macc, m200c) for halo_id, (image, macc, m200c) in enumerate(zip(kSZ_maps, macc, m200c))]
kSZ_halos.sort()

options_macc = np.unique([floor(tSZ_obj.macc) for tSZ_obj in tSZ_halos])
options_m200c = np.unique([tSZ_obj.m200c for tSZ_obj in tSZ_halos])