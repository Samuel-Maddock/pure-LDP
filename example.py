from pure_ldp.frequency_oracles import *
from pure_ldp.heavy_hitters import *

import numpy as np
from collections import Counter

# Super simple synthetic dataset
data = np.concatenate(([1] * 8000, [2] * 4000, [3] * 1000, [4] * 500, [5] * 1000, [6] * 1800, [7] * 2000, [8] * 300))
original_freq = list(Counter(data).values())  # True frequencies of the dataset

# Parameters for experiment
epsilon = 3
d = 8
is_the = True
is_oue = True
is_olh = True

# Optimal Local Hashing (OLH)
client_olh = LHClient(epsilon=epsilon, d=d, use_olh=True)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)

# Optimal Unary Encoding (OUE)
client_oue = UEClient(epsilon=epsilon, d=d, use_oue=True)
server_oue = UEServer(epsilon=epsilon, d=d, use_oue=True)

# Threshold Histogram Encoding (THE)
client_the = HEClient(epsilon=epsilon, d=d)
server_the = HEServer(epsilon=epsilon, d=d, use_the=is_the)

# Hadamard Response (HR)
server_hr = HadamardResponseServer(epsilon, d)
client_hr = HadamardResponseClient(epsilon, d, server_hr.get_hash_funcs())

# Apple's Count Mean Sketch (CMS)
k = 128 # 128 hash functions
m = 1024 # Each hash function maps to the domain {0, ... 1023}

server_cms = CMSServer(epsilon, k, m)
client_cms = CMSClient(epsilon, server_cms.get_hash_funcs(), m)

# Simulate client-side privatisation + server-side aggregation
for item in data:
    priv_olh_data = client_olh.privatise(item)
    priv_oue_data = client_oue.privatise(item)
    priv_the_data = client_the.privatise(item)
    priv_hr_data = client_hr.privatise(item)

    server_olh.aggregate(priv_olh_data)
    server_oue.aggregate(priv_oue_data)
    server_the.aggregate(priv_the_data)
    server_hr.aggregate(priv_hr_data)

# Note instead, we could use server.aggregate_all(list_of_privatised_data) see the Apple CMS example below

# Simulate server-side estimation
oue_estimates = []
olh_estimates = []
the_estimates = []
hr_estimates = []
mse_arr = np.zeros(5)

for i in range(0, d):
    olh_estimates.append(round(server_olh.estimate(i + 1)))
    oue_estimates.append(round(server_oue.estimate(i + 1)))
    the_estimates.append(round(server_the.estimate(i + 1)))
    hr_estimates.append(round(server_hr.estimate(i + 1)))

# Note in the above we could do server.estimate_all(range(1, d+1)) to save looping, see the apple CMS example below


# ------------------------------ Apple CMS Example (using aggregate_all and estimate_all) -------------------------

priv_data = [client_cms.privatise(item) for item in data]
server_cms.aggregate_all(priv_data)
cms_estimates = server_cms.estimate_all(range(1, d+1))

# ------------------------------ Experiment Output (calculating variance) -------------------------

for i in range(0, d):
    mse_arr[0] += (olh_estimates[i] - original_freq[i]) ** 2
    mse_arr[1] += (oue_estimates[i] - original_freq[i]) ** 2
    mse_arr[2] += (the_estimates[i] - original_freq[i]) ** 2
    mse_arr[3] += (hr_estimates[i] - original_freq[i]) ** 2
    mse_arr[4] += (cms_estimates[i] - original_freq[i]) ** 2

mse_arr = mse_arr / d

print("\n")
print("Experiment run on a dataset of size", len(data), "with d=", d, "and epsilon=", epsilon, "\n")
print("Optimised Local Hashing (OLH) Variance: ", mse_arr[0])
print("Optimised Unary Encoding (OUE) Variance: ", mse_arr[1])
print("Threshold Histogram Encoding (THE) Variance: ", mse_arr[2])
print("Hadamard response (HR) Variance:", mse_arr[3])
print(sum(hr_estimates))

print("Apple CMS Variance:", mse_arr[4])
print("\n")
print("Original Frequencies:", original_freq)
print("OLH Estimates:", olh_estimates)
print("OUE Estimates:", oue_estimates)
print("THE Estimates:", the_estimates)
print("HR Estimates:", hr_estimates)
print("CMS Estimates:", cms_estimates)
print("Note: We round estimates to the nearest integer")


# ------------------------------ Heavy Hitters - PEM Simulation -------------------------

pem_client = PEMClient(epsilon=3, start_length=2, max_string_length=6, fragment_length=2)
pem_server = PEMServer(epsilon=3, start_length=2, max_string_length=6, fragment_length=2)

s1 = "101101"
s2 = "111111"
s3 = "100000"
s4 = "101100"

print("\nRunning Prefix Extending Method (PEM) to find heavy hitters")
print("Finding top 3 strings, where the alphabet is:", s1, s2, s3, s4)

data = np.concatenate(([s1] * 8000, [s2] * 4000, [s3] * 1000, [s4] * 500))

for index, item in enumerate(data):
    priv = pem_client.privatise(item)
    pem_server.aggregate(priv)

# Can either specify top-k based or threshold based
    # Threshold of 0.05 means we find any possible heavy hitters that have a frequency >= 5%
    # Top-k of three means we try to find the top-3 most frequent strings

heavy_hitters, frequencies = pem_server.find_heavy_hitters(threshold=0.05)
print("Top strings found are:", heavy_hitters, " with frequencies", frequencies)