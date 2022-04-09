import torch
import numpy as np

from torch.autograd import Variable

import sys

import json
sys.path.append('../../')

import streamlit as st
from streamlit_plotly_events import plotly_events

from eeggan.examples.conv_lin.model import Generator, Discriminator

import plotly.express as px

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


st.set_page_config(layout="wide")
n_samples = 10000
n_gen_samples = 1
n_z = 16

generator = None
cloud = None
pca = None

main_sample = np.array([ 0.00280906,  0.00686481,  0.01777054,  0.01241797, -0.00215334,
        0.00084874, -0.02818346, -0.03494223,  0.07948139,  0.10796523,
        0.07154126,  0.00578119,  0.01263705, -0.02969465, -0.05244173,
       -0.04363361,  0.0027995 ,  0.01090148,  0.0399012 ,  0.02837698,
        0.03967757,  0.03571384,  0.03164138,  0.04042603], dtype=np.float32)

visualization, signals = st.columns(2)


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

with visualization:
    st.markdown('## ERP Latent space representations')

    fig = px.scatter(x=cloud.T[0], y=cloud.T[1])
    selected_points_dict = plotly_events(fig, select_event=True)
    if len(selected_points_dict) > 0:
        selected_points = np.array([[p['x'], p['y']] for p in selected_points_dict])

        selected_z = pca.inverse_transform(selected_points).astype(np.float32)

        z_vars = Variable(torch.from_numpy(selected_z),volatile=True).cpu()

        sample = generator(z_vars).detach().numpy().reshape(len(selected_points), -1).mean(axis=0)
        current_sample = (0.1 * sample + 0.9 * main_sample) / 2
        try:
            previous_sample = np.array(json.load(open('./previous.json')))
        except:
            previous_sample = None

        fig = plt.figure(figsize=(5, 2))

        current_sample = current_sample.repeat(3, axis=0)
        current_sample = savitzky_golay(current_sample, 7, 3)

        plt.plot(np.linspace(-0.2, 0.8, current_sample.shape[0]), current_sample, label='selected')
        if previous_sample is not None:
            plt.plot(np.linspace(-0.2, 0.8, previous_sample.shape[0]), previous_sample, label='previous')
        plt.legend()
        json.dump(current_sample.tolist(), open('./previous.json', 'w'))

        plt.axvline(0, linestyle='--')

        with signals:
            st.markdown('## Selected samples, averaged')
            st.pyplot(fig)