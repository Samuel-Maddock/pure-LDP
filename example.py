from pure_ldp.unary_encoding.ue_client import UEClient
from pure_ldp.unary_encoding.ue_server import UEServer

from pure_ldp.local_hashing.lh_client import LHClient
from pure_ldp.local_hashing.lh_server import LHServer

from pure_ldp.histogram_encoding.he_client import HEClient
from pure_ldp.histogram_encoding.he_server import HEServer

from pure_ldp.prefix_extending.pem_client import PEMClient
from pure_ldp.prefix_extending.pem_server import PEMServer

import numpy as np
import time
from collections import Counter

# Super simple synthetic dataset
data = np.concatenate(([1]*8000, [2]*4000, [3]*1000, [4]*500, [5]*1000, [6]*1800, [7]*2000, [8]*300))
original_freq = list(Counter(data).values()) # True frequencies of the dataset

# Parameters for experiment
epsilon = 3
d = 8
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


# Uncomment to run PEM code

# pem_client = PEMClient(epsilon=3, domain_size=6, start_length=2, segment_length=2)
# pem_server = PEMServer(epsilon=3, domain_size=6, start_length=2, segment_length=2)
#
# s1 = "101101"
# s2 = "111111"
# s3 = "100000"
# s4 = "101100"
#
# data = np.concatenate(([s1]*8000, [s2]*4000, [s3]*1000, [s4]*500))
#
# for index,item in enumerate(data):
#     pem_server.aggregate(*pem_client.privatise(item, index), index)
#
#
# print(pem_server.find_top_k(3))
