from algorithms.unary_encoding.UEClient import UEClient
from algorithms.unary_encoding.UEServer import UEServer

from algorithms.local_hashing.LHClient import LHClient
from algorithms.local_hashing.LHServer import LHServer

from algorithms.histogram_encoding.HEClient import HEClient
from algorithms.histogram_encoding.HEServer import HEServer

import numpy as np
import time
from collections import Counter

# Super simple synthetic dataset
data = np.concatenate(([1]*8000, [2]*4000, [3]*1000, [4]*500))
original_freq = list(Counter(data).values()) # True frequencies of the dataset

# Parameters for experiment
epsilon = 3
d = 4
is_the = True
is_oue = True
is_olh = True

# Optimal Local Hashing (OLH)
client_olh = LHClient(epsilon=epsilon, use_olh=True)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)

# Optimal Unary Encoding (OUE)
client_oue = UEClient(epsilon=epsilon, d=d, use_oue=True)
server_oue = UEServer(epsilon=epsilon, d=d, use_oue=True)

# Threshold Histogram Encoding (THE)
client_the = HEClient(epsilon=epsilon, d=d)
server_the = HEServer(epsilon=epsilon, d=d, use_the=is_the)

# Simulate client-side privatisation + server-side aggregation
for index, item in enumerate(data):
    priv_olh_data = client_olh.privatise(item, index)
    priv_oue_data = client_oue.privatise(item)
    priv_the_data = client_the.privatise(item)

    server_olh.aggregate(priv_olh_data, index)
    server_oue.aggregate(priv_oue_data)
    server_the.aggregate(priv_the_data)

# Simulate server-side estimation
oue_estimates = []
olh_estimates = []
the_estimates = []
mse_arr = np.zeros(3)

for i in range(0, d):
    olh_estimates.append(round(server_olh.estimate(i+1)))
    oue_estimates.append(round(server_oue.estimate(i+1)))
    the_estimates.append(round(server_the.estimate(i+1)))

# Calculate variance
for i in range(0,d):
    mse_arr[0] += (olh_estimates[i] - original_freq[i])**2
    mse_arr[1] += (oue_estimates[i] - original_freq[i])**2
    mse_arr[2] += (the_estimates[i] - original_freq[i])**2

mse_arr = mse_arr/d

# Output:
print("\n")
print("Experiment run on a dataset of size", len(data), "with d=",d, "and epsilon=",epsilon, "\n")
print("Optimised Local Hashing (OLH) Variance: ", mse_arr[0])
print("Optimised Unary Encoding (OUE) Variance: ", mse_arr[1])
print("Threshold Histogram Encoding (THE) Variance: ", mse_arr[2])
print("\n")
print("Original Frequencies:", original_freq)
print("OLH Estimates:", olh_estimates)
print("OUE Estimates:", oue_estimates)
print("THE Estimates:", the_estimates)
print("Note: We round estimates to the nearest integer")
