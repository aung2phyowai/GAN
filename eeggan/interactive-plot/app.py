import torch
import numpy as np

from torch.autograd import Variable

import sys

sys.path.append('../../')

import streamlit as st
from streamlit_plotly_events import plotly_events

from eeggan.examples.conv_lin.model import Generator, Discriminator

import plotly.express as px

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n_samples = 10000
n_gen_samples = 1
n_z = 16

generator = None
cloud = None
pca = None

def initialize_gan():
    generator = Generator(1, n_z)
    generator.train_init()
    generator.load_model('../examples/conv_lin/test.cnt/Progressive0.gen')
    return generator

@st.cache
def initialize_cloud():
    rng = np.random.RandomState(0)
    z_vars_im = rng.normal(0,1,size=(n_samples, n_z)).astype(np.float32)
    pca = PCA(n_components=2)
    cloud = pca.fit_transform(z_vars_im)
    return pca, cloud


generator = initialize_gan()
pca, cloud = initialize_cloud()

main_sample = np.array([ 0.00280906,  0.00686481,  0.01777054,  0.01241797, -0.00215334,
        0.00084874, -0.02818346, -0.03494223,  0.07948139,  0.10796523,
        0.07154126,  0.00578119,  0.01263705, -0.02969465, -0.05244173,
       -0.04363361,  0.0027995 ,  0.01090148,  0.0399012 ,  0.02837698,
        0.03967757,  0.03571384,  0.03164138,  0.04042603], dtype=np.float32)

fig = px.scatter(x=cloud.T[0], y=cloud.T[1])
selected_points_dict = plotly_events(fig, select_event=True)
if len(selected_points_dict) > 0:

    selected_points = np.array([[p['x'], p['y']] for p in selected_points_dict])

    selected_z = pca.inverse_transform(selected_points).astype(np.float32)
    # selected_z = 0.001 * np.random.rand(n_gen_samples, n_z).astype(np.float32) + selected_z

    z_vars = Variable(torch.from_numpy(selected_z),volatile=True).cpu()

    sample = generator(z_vars).detach().numpy().reshape(len(selected_points), -1).mean(axis=0)

    fig = plt.figure(figsize=(5, 2))
    plt.plot(np.linspace(-0.2, 0.8, sample.shape[0]), (0.1 * sample + 0.9 * main_sample) / 2)
    plt.axvline(0, linestyle='--')
    st.pyplot(fig)