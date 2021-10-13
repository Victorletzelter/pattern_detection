#import tensorflow as tf
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import nitime.algorithms as tsa
import matplotlib.mlab as m
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate


import requests
from stockstats import StockDataFrame as sdf

from math import *
from scipy.signal import find_peaks

import copy

import eqsig.single
import eqsig.multiple

import eqsig
from matplotlib import rc

from scipy import signal
from scipy.fft import fftshift

import statsmodels
from statsmodels import graphics
from statsmodels.graphics import gofplots

import cmath
from cmath import *

import random as rd

import statistics as s

def rogne(M) :
    return(M[0:26])
