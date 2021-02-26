from pure_ldp.frequency_oracles import *
from pure_ldp.simulations.helpers.FrequencyOracleSimulation import FrequencyOracleSimulation
from pure_ldp.simulations.helpers.HeavyHitterExperiment import HeavyHitterExperiment
from pure_ldp.core import generate_hash_funcs

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

from collections import Counter, defaultdict, OrderedDict
from pandas import Series


# #-------------------- Parameters for simulation --------------------

# --------- General Parameters -----------
N = 100000
epsilon = 3  # 1, 0.1

# --------- CMS/HCMS Parameters -----------

# Default parameters but these are overidden in simulations...
m = 2048
k = 1024

# --------- RAPPOR -----------
num_bloombits = 128  # Max size is 256 bits
num_hashes = 2  # Recommended to use 2 hashes
num_of_cohorts = 8  # Max cohorts is 64

# PrivCountSketch Parameters
l = 250
numBits = int(math.floor(math.log(N, 2)) + 1)
w = 2 ** numBits
w=2048

# ------------ CMS ------------
cms_params = {"m": m, "k": k, "epsilon": epsilon}
cms = {"client_params": cms_params, "server_params": cms_params}
hcms = copy.deepcopy(cms)
hcms["client_params"]["is_hadamard"] = True
hcms["server_params"]["is_hadamard"] = True

# ----------------------------


def generate_zipf_data(d, n=1000000, large_domain=False, name=None, s=1.1):
    data = np.random.zipf(s, round(N * 50)).astype(int)  # Generate test data
    data_counter = Counter(data)
    keys, freqs = zip(*data_counter.most_common(d))
    key_dict = dict(zip(keys, range(0, len(keys))))

    if large_domain:
        d1 = [key_dict[item] for item in keys]
        d2 = data[np.where(np.in1d(data, keys))]
        d2 = [key_dict[item] for item in d2]
        d2 = np.random.choice(d2, size=n - len(d1))
        data = np.concatenate([d1, d2])
    else:
        data = data[np.where(np.in1d(data, keys))]
        data = [key_dict[item] for item in data]
        data = np.random.choice(data, size=n)

    print("Total Size of Dataset:", len(data))
    print("Total unique items:", len(Counter(data).keys()))
    print("Max Frequency: ", max(Counter(data).values()))
    # print(Counter(data))
    # plt.hist(x=data, bins=1000)
    # plt.show()
    if name is not None:
        np.save("./data/" + name, data)
    return data

# ----------------------------
# GROUP 1 EXPERIMENTS
# ----------------------------

# Figure 1a
def group1_vary_eps():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", calc_top_k=top_k, display_full_stats=True, autosave=True)
    repeats = 5
    epsilons = np.arange(0.5, 5.5, 0.5)

    de_client_params = {"epsilon": 3}
    de_server_params = {"epsilon": 3}

    sue_client_params = copy.deepcopy(de_client_params)
    sue_server_params = copy.deepcopy(de_server_params)
    sue_client_params["use_oue"] = False
    sue_server_params["use_oue"] = False

    oue_client_params = copy.deepcopy(de_client_params)
    oue_server_params = copy.deepcopy(de_server_params)
    oue_client_params["use_oue"] = True
    oue_server_params["use_oue"] = True

    DE = {"client_params": de_client_params, "server_params": de_server_params}
    SUE = {"client_params": sue_client_params, "server_params": sue_server_params}
    OUE = {"client_params": oue_client_params, "server_params": oue_server_params}

    experiment_list = []

    data = generate_zipf_data(d=1024, n=100000)
    print(len(data))
    print(len(Counter(data).keys()))

    for eps in epsilons:
        for i in range(0, repeats):
            de_params = copy.deepcopy(DE)
            sue_params = copy.deepcopy(SUE)
            oue_params = copy.deepcopy(OUE)
            freq_oracles = [de_params, sue_params, oue_params]

            for fo in freq_oracles:
                fo["data"] = data
                fo["client_params"]["epsilon"] = eps
                fo["server_params"]["epsilon"] = eps

            experiment_list.append((("DE", "e =" + str(eps)), de_params))
            experiment_list.append((("UE", "SUE", "e =" + str(eps)), sue_params))
            experiment_list.append((("UE", "OUE", "e =" + str(eps)), oue_params))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 1b
def group1_vary_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", calc_top_k=top_k, display_full_stats=True, autosave=True)
    repeats = 5

    de_client_params = {"epsilon": 3}
    de_server_params = {"epsilon": 3}

    sue_client_params = copy.deepcopy(de_client_params)
    sue_server_params = copy.deepcopy(de_server_params)
    sue_client_params["use_oue"] = False
    sue_server_params["use_oue"] = False

    oue_client_params = copy.deepcopy(de_client_params)
    oue_server_params = copy.deepcopy(de_server_params)
    oue_client_params["use_oue"] = True
    oue_server_params["use_oue"] = True

    DE = {"client_params": de_client_params, "server_params": de_server_params}
    SUE = {"client_params": sue_client_params, "server_params": sue_server_params}
    OUE = {"client_params": oue_client_params, "server_params": oue_server_params}

    experiment_list = []

    data_list = []
    for i in range(0, 10):
        data_list.append(generate_zipf_data(d=2 ** (i + 2), n=100000))

    for i, data in enumerate(data_list):
        for j in range(0, repeats):
            de_params = copy.deepcopy(DE)
            sue_params = copy.deepcopy(SUE)
            oue_params = copy.deepcopy(OUE)
            freq_oracles = [de_params, sue_params, oue_params]

            for fo in freq_oracles:
                fo["data"] = data

            experiment_list.append((("DE", "d =" + str(2 ** (i + 2))), de_params))
            experiment_list.append((("UE", "SUE", "d =" + str(2 ** (i + 2))), sue_params))
            experiment_list.append((("UE", "OUE", "d =" + str(2 ** (i + 2))), oue_params))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

def group1_vary_n():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", calc_top_k=top_k, display_full_stats=True, autosave=True)
    repeats = 5

    de_client_params = {"epsilon": 3}
    de_server_params = {"epsilon": 3}

    sue_client_params = copy.deepcopy(de_client_params)
    sue_server_params = copy.deepcopy(de_server_params)
    sue_client_params["use_oue"] = False
    sue_server_params["use_oue"] = False

    oue_client_params = copy.deepcopy(de_client_params)
    oue_server_params = copy.deepcopy(de_server_params)
    oue_client_params["use_oue"] = True
    oue_server_params["use_oue"] = True

    DE = {"client_params": de_client_params, "server_params": de_server_params}
    SUE = {"client_params": sue_client_params, "server_params": sue_server_params}
    OUE = {"client_params": oue_client_params, "server_params": oue_server_params}

    experiment_list = []
    n_list = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    data_list = []
    for n in n_list:
        data_list.append(generate_zipf_data(d=128, n=n))

    for i, data in enumerate(data_list):
        for j in range(0, repeats):
            de_params = copy.deepcopy(DE)
            sue_params = copy.deepcopy(SUE)
            oue_params = copy.deepcopy(OUE)
            freq_oracles = [de_params, sue_params, oue_params]

            for fo in freq_oracles:
                fo["data"] = data

            experiment_list.append((("DE", "n =" + str(n_list[i])), de_params))
            experiment_list.append((("UE", "SUE", "n =" + str(n_list[i])), sue_params))
            experiment_list.append((("UE", "OUE", "n =" + str(n_list[i])), oue_params))

    simulation.run_and_plot(experiment_list, display_stats_only=True)


# ----------------------------
# GROUP 2 EXPERIMENTS
# ----------------------------

# Figure 3a
def group2_vary_eps():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 5
    epsilons = np.arange(0.5, 5.5, 0.5)
    k_list = [100, 500, 1000, 5000, 10000]

    blh_client_params = {"epsilon": 3, "use_olh": False, "g": 2}
    blh_server_params = {"epsilon": 3, "use_olh": False, "g": 2}

    olh_client_params = copy.deepcopy(blh_client_params)
    olh_server_params = copy.deepcopy(blh_server_params)
    olh_client_params["use_olh"] = True
    olh_server_params["use_olh"] = True

    flh_client_params = copy.deepcopy(olh_client_params)
    flh_server_params = copy.deepcopy(olh_server_params)
    flh_client_params["k"] = 1000
    flh_server_params["k"] = 1000

    BLH = {"client_params": blh_client_params, "server_params": blh_server_params}
    OLH = {"client_params": olh_client_params, "server_params": olh_server_params}
    FLH = {"client_params": flh_client_params, "server_params": flh_server_params}

    experiment_list = []

    data = generate_zipf_data(d=1024, n=100000)
    print(len(data))
    print(len(Counter(data).keys()))
    for eps in epsilons:
        for i in range(0, repeats):
            blh_params = copy.deepcopy(BLH)
            olh_params = copy.deepcopy(OLH)
            flh_params = copy.deepcopy(FLH)
            freq_oracles = [blh_params, olh_params, flh_params]

            for fo in freq_oracles:
                fo["data"] = data
                fo["client_params"]["epsilon"] = eps
                fo["server_params"]["epsilon"] = eps

            experiment_list.append((("LH", "BLH", "e =" + str(eps)), blh_params))
            experiment_list.append((("LH", "OLH", "e =" + str(eps)), olh_params))

            for k in k_list:
                flh_params["client_params"]["k"] = k
                flh_params["server_params"]["k"] = k
                experiment_list.append((("FastLH", "e =" + str(eps) + " k=" + str(k)), copy.deepcopy(flh_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 3b
def group2_vary_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", calc_top_k=top_k, display_full_stats=True, autosave=True)
    repeats = 5
    k_list = [100, 500, 1000, 5000, 10000]

    blh_client_params = {"epsilon": 3, "use_olh": False, "g": 2}
    blh_server_params = {"epsilon": 3, "use_olh": False, "g": 2}

    olh_client_params = copy.deepcopy(blh_client_params)
    olh_server_params = copy.deepcopy(blh_server_params)
    olh_client_params["use_olh"] = True
    olh_server_params["use_olh"] = True

    flh_client_params = copy.deepcopy(olh_client_params)
    flh_server_params = copy.deepcopy(olh_server_params)
    flh_client_params["k"] = 100
    flh_server_params["k"] = 100

    BLH = {"client_params": blh_client_params, "server_params": blh_server_params}
    OLH = {"client_params": olh_client_params, "server_params": olh_server_params}
    FLH = {"client_params": flh_client_params, "server_params": flh_server_params}

    experiment_list = []

    data_list = []
    for i in range(0, 10):
        data_list.append(generate_zipf_data(d=2 ** (i + 2), n=100000))

    for i, data in enumerate(data_list):
        for j in range(0, repeats):
            blh_params = copy.deepcopy(BLH)
            olh_params = copy.deepcopy(OLH)
            flh_params = copy.deepcopy(FLH)
            freq_oracles = [blh_params, olh_params, flh_params]

            for fo in freq_oracles:
                fo["data"] = data

            experiment_list.append((("LH", "BLH", "d =" + str(2 ** (i + 2))), blh_params))
            experiment_list.append((("LH", "OLH", "d =" + str(2 ** (i + 2))), olh_params))
            for k in k_list:
                flh_params["client_params"]["k"] = k
                flh_params["server_params"]["k"] = k
                experiment_list.append(
                    (("FastLH", "d =" + str(2 ** (i + 2)) + " k=" + str(k)), copy.deepcopy(flh_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)


def group2_vary_n():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", calc_top_k=top_k, display_full_stats=True, autosave=True)
    repeats = 5
    k_list = [100, 500, 1000, 5000, 10000]

    blh_client_params = {"epsilon": 3, "use_olh": False, "g": 2}
    blh_server_params = {"epsilon": 3, "use_olh": False, "g": 2}

    olh_client_params = copy.deepcopy(blh_client_params)
    olh_server_params = copy.deepcopy(blh_server_params)
    olh_client_params["use_olh"] = True
    olh_server_params["use_olh"] = True

    flh_client_params = copy.deepcopy(olh_client_params)
    flh_server_params = copy.deepcopy(olh_server_params)
    flh_client_params["k"] = 100
    flh_server_params["k"] = 100

    BLH = {"client_params": blh_client_params, "server_params": blh_server_params}
    OLH = {"client_params": olh_client_params, "server_params": olh_server_params}
    FLH = {"client_params": flh_client_params, "server_params": flh_server_params}

    experiment_list = []

    n_list = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    data_list = []
    for n in n_list:
        data_list.append(generate_zipf_data(d=128, n=n))

    for i, data in enumerate(data_list):
        for j in range(0, repeats):
            blh_params = copy.deepcopy(BLH)
            olh_params = copy.deepcopy(OLH)
            flh_params = copy.deepcopy(FLH)
            freq_oracles = [blh_params, olh_params, flh_params]

            for fo in freq_oracles:
                fo["data"] = data

            experiment_list.append((("LH", "BLH", "n =" + str(n_list[i])), blh_params))
            experiment_list.append((("LH", "OLH", "n =" + str(n_list[i])), olh_params))
            for k in k_list:
                flh_params["client_params"]["k"] = k
                flh_params["server_params"]["k"] = k
                experiment_list.append((("FastLH", "n =" + str(n_list[i]) + " k=" + str(k)), copy.deepcopy(flh_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# ----------------------------
# GROUP 3 EXPERIMENTS
# ----------------------------

# Figure 4b
def group3_vary_eps():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 5
    epsilons = np.arange(0.5, 5.5, 0.5)
    t_range = [1, 2, 3, 4, 5]

    hr_client_params = {"epsilon": 3}
    hr_server_params = {"epsilon": 3}

    hm_client_params = copy.deepcopy(hr_client_params)
    hm_server_params = copy.deepcopy(hr_server_params)
    hm_client_params["t"] = 1
    hm_server_params["t"] = 1

    HR = {"client_params": hr_client_params, "server_params": hr_server_params}
    HM = {"client_params": hm_client_params, "server_params": hm_server_params}

    experiment_list = []

    data = generate_zipf_data(d=1024, n=100000)
    print(len(data))
    print(len(Counter(data).keys()))
    for eps in epsilons:
        for i in range(0, repeats):
            hr_params = copy.deepcopy(HR)
            hm_params = copy.deepcopy(HM)

            freq_oracles = [hr_params, hm_params]

            for fo in freq_oracles:
                fo["data"] = data
                fo["client_params"]["epsilon"] = eps
                fo["server_params"]["epsilon"] = eps

            experiment_list.append((("HR", "e=" + str(eps)), hr_params))

            for t in t_range:
                hm_params["client_params"]["t"] = t
                hm_params["server_params"]["t"] = t
                experiment_list.append((("HadamardMech", "e=" + str(eps) + " t=" + str(t)), copy.deepcopy(hm_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 4c
def group3_vary_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 5
    t_range = [1, 2, 3, 4, 5]

    hr_client_params = {"epsilon": 3}
    hr_server_params = {"epsilon": 3}

    hm_client_params = copy.deepcopy(hr_client_params)
    hm_server_params = copy.deepcopy(hr_server_params)
    hm_client_params["t"] = 1
    hm_server_params["t"] = 1

    HR = {"client_params": hr_client_params, "server_params": hr_server_params}
    HM = {"client_params": hm_client_params, "server_params": hm_server_params}

    experiment_list = []

    data_list = []
    for i in range(0, 10):
        data_list.append(generate_zipf_data(d=2 ** (i + 2), n=100000))

    for j, data in enumerate(data_list):
        print(Counter(data))
        for i in range(0, repeats):
            hr_params = copy.deepcopy(HR)
            hm_params = copy.deepcopy(HM)

            freq_oracles = [hr_params, hm_params]

            for fo in freq_oracles:
                fo["data"] = data

            experiment_list.append((("HR", "d =" + str(2 ** (j + 2))), hr_params))
            for t in t_range:
                hm_params["client_params"]["t"] = t
                hm_params["server_params"]["t"] = t
                experiment_list.append(
                    (("HadamardMech", "d=" + str(2 ** (j + 2)) + " t=" + str(t)), copy.deepcopy(hm_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 4a
def group3_vary_t():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 10
    epsilon = math.log(3)

    hr_client_params = {"epsilon": epsilon}
    hr_server_params = {"epsilon": epsilon}

    hm_client_params = copy.deepcopy(hr_client_params)
    hm_server_params = copy.deepcopy(hr_server_params)
    hm_client_params["t"] = 1
    hm_server_params["t"] = 1

    HR = {"client_params": hr_client_params, "server_params": hr_server_params}
    HM = {"client_params": hm_client_params, "server_params": hm_server_params}

    experiment_list = []

    data = generate_zipf_data(d=1024, n=100000)

    t_range = [1, 2, 3, 4, 5, 6, 7]
    eps_range = [0.1, 0.5, 1, 2, 3, 4]
    for eps in eps_range:
        for i in range(0, repeats):
            hr_params = copy.deepcopy(HR)
            hm_params = copy.deepcopy(HM)

            freq_oracles = [hr_params, hm_params]

            for fo in freq_oracles:
                fo["data"] = data
                hm_params["server_params"]["epsilon"] = eps
                hm_params["client_params"]["epsilon"] = eps

            for t in t_range:
                hm_params["server_params"]["t"] = t
                hm_params["client_params"]["t"] = t
                experiment_list.append((("HadamardMech", "e=" + str(eps) + " t=" + str(t)), copy.deepcopy(hm_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)


def group3_high_privacy():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 5

    hr_client_params = {"epsilon": 3}
    hr_server_params = {"epsilon": 3}

    hm_client_params = copy.deepcopy(hr_client_params)
    hm_server_params = copy.deepcopy(hr_server_params)
    hm_client_params["t"] = 1
    hm_server_params["t"] = 1

    HR = {"client_params": hr_client_params, "server_params": hr_server_params}
    HM = {"client_params": hm_client_params, "server_params": hm_server_params}

    experiment_list = []
    data = generate_zipf_data(d=2048, n=100000, large_domain=True)
    e = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for j, eps in enumerate(e):
        for i in range(0, repeats):
            hr_params = copy.deepcopy(HR)
            hm_params = copy.deepcopy(HM)

            freq_oracles = [hr_params, hm_params]

            for fo in freq_oracles:
                fo["server_params"]["epsilon"] = eps
                fo["client_params"]["epsilon"] = eps
                fo["data"] = data

            experiment_list.append((("HR", "e =" + str(e[j])), copy.deepcopy(hr_params)))
            experiment_list.append((("HadamardMech", "e=" + str(e[j])), copy.deepcopy(hm_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# ----------------------------
# GROUP 5 EXPERIMENTS
# ----------------------------

# Figure 8
def group5_bloom_comparison():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)

    d_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    data_list = []
    for d in d_list:
        data_list.append(generate_zipf_data(d=d, n=1000000, large_domain=True))

    k = 16
    m = 128
    repeats = 5
    eps = 3
    experiment_list = []
    sketch_est_types = ["(Min)", "(Median)", "(Mean)", "(Debias Mean)"]
    sketch_types = [False, True]
    rappor = {"server_params": {"f": 0.64, "m": 128, "k": 2}, "client_params": {"f": 0.64, "m": 128}}
    reg_const = [0.005]

    for i in range(0, repeats):
        for j, data in enumerate(data_list):
            for const in reg_const:
                rappor["server_params"]["reg_const"] = const
                rappor["data"] = data
                experiment_list.append((("rappor", "rappor reg=" + str(const),
                                         "d=" + str(d)), copy.deepcopy(rappor)))

            for j, sketch_est_type in enumerate(sketch_est_types):
                for sketch_type in sketch_types:
                    sr = {"server_params": {}, "client_params": {}}
                    sr["server_params"]["m"] = m
                    sr["client_params"]["m"] = m
                    sr["server_params"]["k"] = k
                    sr["client_params"]["k"] = k
                    sr["client_params"]["epsilon"] = eps
                    sr["server_params"]["epsilon"] = eps
                    sr["server_params"]["sketch_method"] = j
                    sr["server_params"]["count_sketch"] = sketch_type
                    sr["client_params"]["count_sketch"] = sketch_type
                    sr["data"] = data

                    # SR with DE
                    SR_de = copy.deepcopy(sr)
                    SR_de["server_params"]["fo_server"] = DEServer(epsilon=eps, d=m)
                    SR_de["client_params"]["fo_client"] = DEClient(epsilon=eps, d=m)

                    if sketch_type:
                        name = " CS "
                    else:
                        name = " "
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with DE ",
                                             "d=" + str(d)), copy.deepcopy(SR_de)))

    print("Total length of experiments:", len(experiment_list))
    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 6
def group5_bloom_reg():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)

    d_list = [100000]
    data_list = []
    for d in d_list:
        data_list.append(generate_zipf_data(d=d, n=1000000, large_domain=True))

    repeats = 5
    eps = 3
    experiment_list = []
    rappor = {"server_params": {"f": 0.64, "m": 128, "k": 2}, "client_params": {"f": 0.64, "m": 128}}
    reg_const = [0, 0.0001, 0.0005, 0.00075, 0.001, 0.005, 0.0075, 0.01, 0.05, 0.075, 0.1, 0.5, 0.75, 1, 5]
    bloom = [32, 64, 128]
    for i in range(0, repeats):
        for j, data in enumerate(data_list):
            for m in bloom:
                for const in reg_const:
                    rappor["server_params"]["reg_const"] = const
                    rappor["data"] = data
                    rappor["server_params"]["m"] = m
                    rappor["client_params"]["m"] = m
                    experiment_list.append((("rappor", "rappor reg=" + str(const),
                                             "d=" + str(d) + " m=" + str(m)), copy.deepcopy(rappor)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)


# Figure 7
def group5_SR_vary_m():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    data = generate_zipf_data(d=100000, n=1000000, large_domain=True)
    k = 32
    m_list = [32, 64, 128, 256, 512, 1024, 2048]
    repeats = 5
    epsilons = [3]
    experiment_list = []
    sketch_est_types = ["(Min)", "(Median)", "(Mean)", "(Debias Mean)"]
    sketch_types = [False, True]

    for i in range(0, repeats):
        for eps in epsilons:
            for m in m_list:
                for j, sketch_est_type in enumerate(sketch_est_types):
                    for sketch_type in sketch_types:
                        sr = {"server_params": {}, "client_params": {}}
                        sr["server_params"]["m"] = m
                        sr["client_params"]["m"] = m
                        sr["server_params"]["k"] = k
                        sr["client_params"]["k"] = k
                        sr["client_params"]["epsilon"] = eps
                        sr["server_params"]["epsilon"] = eps
                        sr["server_params"]["sketch_method"] = j
                        sr["server_params"]["count_sketch"] = sketch_type
                        sr["client_params"]["count_sketch"] = sketch_type
                        sr["data"] = data

                        # SR with FLH
                        SR_flh = copy.deepcopy(sr)
                        SR_flh["server_params"]["lh_k"] = 500
                        SR_flh["client_params"]["lh_k"] = 500

                        if sketch_type:
                            name = " CS "
                        else:
                            name = " "
                        experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with FLH ",
                                                 "m=" + str(m)), copy.deepcopy(SR_flh)))
    print("Total Number of Experiments:", len(experiment_list))
    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 7
def group5_SR_vary_k():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    data = generate_zipf_data(d=100000, n=1000000, large_domain=True)
    m = 1024
    k_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    repeats = 5
    epsilons = [3]
    experiment_list = []
    sketch_est_types = ["(Min)", "(Median)", "(Mean)", "(Debias Mean)"]
    sketch_types = [False, True]

    for i in range(0, repeats):
        for eps in epsilons:
            for k in k_list:
                for j, sketch_est_type in enumerate(sketch_est_types):
                    for sketch_type in sketch_types:
                        sr = {"server_params": {}, "client_params": {}}
                        sr["server_params"]["m"] = m
                        sr["client_params"]["m"] = m
                        sr["server_params"]["k"] = k
                        sr["client_params"]["k"] = k
                        sr["client_params"]["epsilon"] = eps
                        sr["server_params"]["epsilon"] = eps
                        sr["server_params"]["sketch_method"] = j
                        sr["server_params"]["count_sketch"] = sketch_type
                        sr["client_params"]["count_sketch"] = sketch_type
                        sr["data"] = data

                        # SR with FLH
                        SR_flh = copy.deepcopy(sr)
                        SR_flh["server_params"]["lh_k"] = 500
                        SR_flh["client_params"]["lh_k"] = 500

                        if sketch_type:
                            name = " CS "
                        else:
                            name = " "
                        experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with FLH ",
                                                 "k=" + str(k)), copy.deepcopy(SR_flh)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 9/10

def group5_ALL_vary_eps():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    k = 32
    m = 1024
    repeats = 5
    epsilons = np.arange(0.5, 5.5, 0.5)
    data = generate_zipf_data(d=100000, n=1000000, large_domain=True)
    experiment_list = []

    sketch_est_types = ["(Min)", "(Median)", "(Mean)", "(Debias Mean)"]
    sketch_types = [False]
    index_mapper = lambda x: x

    for i in range(0, repeats):
        for e in epsilons:
            for j, sketch_est_type in enumerate(sketch_est_types):
                for sketch_type in sketch_types:
                    sr = {"server_params": {}, "client_params": {}}
                    sr["server_params"]["m"] = m
                    sr["client_params"]["m"] = m
                    sr["server_params"]["k"] = k
                    sr["client_params"]["k"] = k
                    sr["client_params"]["epsilon"] = e
                    sr["server_params"]["epsilon"] = e
                    sr["server_params"]["sketch_method"] = j
                    sr["server_params"]["count_sketch"] = sketch_type
                    sr["client_params"]["count_sketch"] = sketch_type
                    sr["data"] = data

                    hr_client_params = {"epsilon": e}
                    hr_server_params = {"epsilon": e}

                    hm_t3_client_params = copy.deepcopy(hr_client_params)
                    hm_t1_client_params = copy.deepcopy(hr_client_params)

                    hm_t3_server_params = copy.deepcopy(hr_server_params)
                    hm_t1_server_params = copy.deepcopy(hr_server_params)

                    t = math.ceil(e) # Optimal t value
                    hm_t3_client_params["t"] = t
                    hm_t3_server_params["t"] = t

                    t=1
                    hm_t1_client_params["t"] = t
                    hm_t1_server_params["t"] = t

                    HM_t3 = {"client_params": hm_t3_client_params, "server_params": hm_t3_server_params}
                    HM_t1 = {"client_params": hm_t1_client_params, "server_params": hm_t1_server_params}

                    # SR with HM
                    SR_hm_t3 = copy.deepcopy(sr)
                    SR_hm_t3["server_params"]["fo_server"] = HadamardMechServer(**HM_t3["server_params"], d=m, index_mapper=index_mapper)
                    SR_hm_t3["client_params"]["fo_client"] = HadamardMechClient(**HM_t3["client_params"], d=m, index_mapper=index_mapper)

                    SR_hm_t1 = copy.deepcopy(SR_hm_t3)
                    SR_hm_t1["server_params"]["fo_server"] = HadamardMechServer(**HM_t1["server_params"], d=m, index_mapper=index_mapper)
                    SR_hm_t1["client_params"]["fo_client"] = HadamardMechClient(**HM_t1["client_params"], d=m, index_mapper=index_mapper)

                    HR = {"client_params": hr_client_params, "server_params": hr_server_params}

                    # SR with FLH
                    SR_flh = copy.deepcopy(sr)
                    SR_flh["server_params"]["lh_k"] = 500
                    SR_flh["client_params"]["lh_k"] = 500

                    # SR with HR
                    SR_hr = copy.deepcopy(sr)
                    hr_server = HadamardResponseServer(**HR["server_params"], d=m, index_mapper=index_mapper)
                    SR_hr["server_params"]["fo_server"] = hr_server
                    SR_hr["client_params"]["fo_client"] = HadamardResponseClient(**HR["client_params"],
                                                                                 hash_funcs=hr_server.get_hash_funcs(),
                                                                                 d=m, index_mapper=index_mapper)
                    if sketch_type:
                        name = " CS "
                    else:
                        name = " "
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with FLH ",
                                             "e=" + str(e)), copy.deepcopy(SR_flh)))
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with HR ",
                                             "e=" + str(e)), copy.deepcopy(SR_hr)))
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with HM t=1",
                                             "e=" + str(e)), copy.deepcopy(SR_hm_t1)))
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with HM t=optimal",
                                             "e=" + str(e)), copy.deepcopy(SR_hm_t3)))

    for i in range(0, repeats):
        for e in epsilons:
            cms_params = copy.deepcopy(cms)
            hcms_params = copy.deepcopy(hcms)

            freq_oracles = [cms_params, hcms_params]

            for fo in freq_oracles:
                fo["server_params"]["m"] = m
                fo["client_params"]["m"] = m
                fo["server_params"]["k"] = k
                fo["client_params"]["k"] = k
                fo["client_params"]["epsilon"] = e
                fo["server_params"]["epsilon"] = e

                fo["data"] = data

            experiment_list.append((("cms", "e=" + str(e)), copy.deepcopy(cms_params)))
            experiment_list.append((("cms", "hcms", "e=" + str(e)), copy.deepcopy(hcms_params)))

    print("Total Number of Experiments:", len(experiment_list))
    simulation.run_and_plot(experiment_list, display_stats_only=True)

def group5_ALL_vary_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    k = 32
    m = 1024
    repeats = 5
    e = 3
    experiment_list = []

    d_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    data_list = []
    for d in d_list:
        data_list.append(generate_zipf_data(d=d, n=1000000, large_domain=True))

    index_mapper = lambda x:x
    sketch_est_types = ["(Min)", "(Median)", "(Mean)", "(Debias Mean)"]
    sketch_types = [False]
    for _ in range(0, repeats):
        for i, data in enumerate(data_list):
            for j, sketch_est_type in enumerate(sketch_est_types):
                for sketch_type in sketch_types:
                    sr = {"server_params": {}, "client_params": {}}
                    sr["server_params"]["m"] = m
                    sr["client_params"]["m"] = m
                    sr["server_params"]["k"] = k
                    sr["client_params"]["k"] = k
                    sr["client_params"]["epsilon"] = e
                    sr["server_params"]["epsilon"] = e
                    sr["server_params"]["sketch_method"] = j
                    sr["server_params"]["count_sketch"] = sketch_type
                    sr["client_params"]["count_sketch"] = sketch_type
                    sr["data"] = data

                    hr_client_params = {"epsilon": e}
                    hr_server_params = {"epsilon": e}

                    hm_t3_client_params = copy.deepcopy(hr_client_params)
                    hm_t1_client_params = copy.deepcopy(hr_client_params)

                    hm_t3_server_params = copy.deepcopy(hr_server_params)
                    hm_t1_server_params = copy.deepcopy(hr_server_params)

                    t = 3
                    hm_t3_client_params["t"] = t
                    hm_t3_server_params["t"] = t

                    t=1
                    hm_t1_client_params["t"] = t
                    hm_t1_server_params["t"] = t

                    HM_t3 = {"client_params": hm_t3_client_params, "server_params": hm_t3_server_params}
                    HM_t1 = {"client_params": hm_t1_client_params, "server_params": hm_t1_server_params}

                    # SR with HM
                    SR_hm_t3 = copy.deepcopy(sr)
                    SR_hm_t3["server_params"]["fo_server"] = HadamardMechServer(**HM_t3["server_params"], d=m, index_mapper=index_mapper)
                    SR_hm_t3["client_params"]["fo_client"] = HadamardMechClient(**HM_t3["client_params"], d=m, index_mapper=index_mapper)

                    SR_hm_t1 = copy.deepcopy(SR_hm_t3)
                    SR_hm_t1["server_params"]["fo_server"] = HadamardMechServer(**HM_t1["server_params"], d=m, index_mapper=index_mapper)
                    SR_hm_t1["client_params"]["fo_client"] = HadamardMechClient(**HM_t1["client_params"], d=m, index_mapper=index_mapper)

                    HR = {"client_params": hr_client_params, "server_params": hr_server_params}

                    # SR with FLH
                    SR_flh = copy.deepcopy(sr)
                    SR_flh["server_params"]["lh_k"] = 500
                    SR_flh["client_params"]["lh_k"] = 500

                    # SR with HR
                    SR_hr = copy.deepcopy(sr)
                    hr_server = HadamardResponseServer(**HR["server_params"], d=m, index_mapper=index_mapper)
                    SR_hr["server_params"]["fo_server"] = hr_server
                    SR_hr["client_params"]["fo_client"] = HadamardResponseClient(**HR["client_params"],
                                                                                 hash_funcs=hr_server.get_hash_funcs(),d=m, index_mapper=index_mapper)

                    if sketch_type:
                        name = " CS "
                    else:
                        name = " "

                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with FLH ",
                                             "d=" + str(d_list[i])), copy.deepcopy(SR_flh)))
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with HR ",
                                             "d=" + str(d_list[i])), copy.deepcopy(SR_hr)))
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with HM t=1",
                                             "d=" + str(d_list[i])), copy.deepcopy(SR_hm_t1)))
                    experiment_list.append((("SketchResponse", "SR" + name + sketch_est_type + " with HM t=3",
                                             "d=" + str(d_list[i])), copy.deepcopy(SR_hm_t3)))

    for i in range(0, repeats):
        for i, data in enumerate(data_list):
            cms_params = copy.deepcopy(cms)
            hcms_params = copy.deepcopy(hcms)

            freq_oracles = [cms_params, hcms_params]

            for fo in freq_oracles:
                fo["server_params"]["m"] = m
                fo["client_params"]["m"] = m
                fo["server_params"]["k"] = k
                fo["client_params"]["k"] = k
                fo["client_params"]["epsilon"] = e
                fo["server_params"]["epsilon"] = e

                fo["data"] = data

            experiment_list.append((("cms", "d=" + str(d_list[i])), copy.deepcopy(cms_params)))
            experiment_list.append((("cms", "hcms", "d=" + str(d_list[i])), copy.deepcopy(hcms_params)))

    print("Total Number of Experiments:", len(experiment_list))
    simulation.run_and_plot(experiment_list, display_stats_only=True)


# Not used...
def group5_apple_vary_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)

    k = 32
    m = 1024
    repeats = 5
    eps = 3

    d_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    data_list = []
    for d in d_list:
        data_list.append(generate_zipf_data(d=d, n=1000000, large_domain=True))

    experiment_list = []
    for i in range(0, repeats):
        for i, data in enumerate(data_list):
            cms_params = copy.deepcopy(cms)
            hcms_params = copy.deepcopy(hcms)

            freq_oracles = [cms_params, hcms_params]

            for fo in freq_oracles:
                fo["server_params"]["m"] = m
                fo["client_params"]["m"] = m
                fo["server_params"]["k"] = k
                fo["client_params"]["k"] = k
                fo["client_params"]["epsilon"] = eps
                fo["server_params"]["epsilon"] = eps

                fo["data"] = data

            experiment_list.append((("cms", "d=" + str(d_list[i])), copy.deepcopy(cms_params)))
            experiment_list.append((("cms", "hcms", "d=" + str(d_list[i])), copy.deepcopy(hcms_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

def group5_apple_vary_eps():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    k = 32
    m = 1024
    repeats = 5

    epsilons = np.arange(0.5, 5.5, 0.5)
    data = generate_zipf_data(d=100000, n=1000000, large_domain=True)

    experiment_list = []
    for i in range(0, repeats):
        for e in epsilons:
            cms_params = copy.deepcopy(cms)
            hcms_params = copy.deepcopy(hcms)

            freq_oracles = [cms_params, hcms_params]

            for fo in freq_oracles:
                fo["server_params"]["m"] = m
                fo["client_params"]["m"] = m
                fo["server_params"]["k"] = k
                fo["client_params"]["k"] = k
                fo["client_params"]["epsilon"] = e
                fo["server_params"]["epsilon"] = e

                fo["data"] = data

            experiment_list.append((("cms", "e=" + str(e)), copy.deepcopy(cms_params)))
            experiment_list.append((("cms", "hcms", "e=" + str(e)), copy.deepcopy(hcms_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

def group5_vary_k():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    data = generate_zipf_data(d=100000, n=1000000, large_domain=True)
    m = 1024
    k_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    repeats = 5
    epsilons = [0.5, 1, 3]

    experiment_list = []
    for i in range(0, repeats):
        for eps in epsilons:
            for k in k_list:
                cms_params = copy.deepcopy(cms)
                hcms_params = copy.deepcopy(hcms)

                freq_oracles = [cms_params, hcms_params]

                for fo in freq_oracles:
                    fo["server_params"]["m"] = m
                    fo["client_params"]["m"] = m
                    fo["server_params"]["k"] = k
                    fo["client_params"]["k"] = k
                    fo["client_params"]["epsilon"] = eps
                    fo["server_params"]["epsilon"] = eps

                    fo["data"] = data

                experiment_list.append((("cms", "k=" + str(k) + " e=" + str(eps)), copy.deepcopy(cms_params)))
                experiment_list.append((("cms", "hcms", "k=" + str(k) + " e=" + str(eps)), copy.deepcopy(hcms_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

def group5_vary_m():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    data = generate_zipf_data(d=100000, n=1000000, large_domain=True)
    k = 32
    m_list = [32, 64, 128, 256, 512, 1024, 2048]
    repeats = 5
    epsilons = [0.5, 1, 3]
    experiment_list = []
    for i in range(0, repeats):
        for eps in epsilons:
            for m in m_list:
                cms_params = copy.deepcopy(cms)
                hcms_params = copy.deepcopy(hcms)

                freq_oracles = [cms_params, hcms_params]

                for fo in freq_oracles:
                    fo["server_params"]["m"] = m
                    fo["client_params"]["m"] = m
                    fo["server_params"]["k"] = k
                    fo["client_params"]["k"] = k
                    fo["client_params"]["epsilon"] = eps
                    fo["server_params"]["epsilon"] = eps

                    fo["data"] = data

                experiment_list.append((("cms", "m=" + str(m) + " e=" + str(eps)), copy.deepcopy(cms_params)))
                experiment_list.append((("cms", "hcms", "m=" + str(m) + " e=" + str(eps)), copy.deepcopy(hcms_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# ----------------------------
# GROUP 6 - Normalisation
# ----------------------------

def group6_normalise_fo_low_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 5

    data_list = []
    for i in range(0, 10):
        data_list.append(generate_zipf_data(d=2 ** (i + 2), n=100000, large_domain=True))

    blh_client_params = {"epsilon": 3, "use_olh": False, "g": 2}
    blh_server_params = {"epsilon": 3, "use_olh": False, "g": 2}

    olh_client_params = copy.deepcopy(blh_client_params)
    olh_server_params = copy.deepcopy(blh_server_params)
    olh_client_params["use_olh"] = True
    olh_server_params["use_olh"] = True

    flh_client_params = copy.deepcopy(olh_client_params)
    flh_server_params = copy.deepcopy(olh_server_params)
    FLH = {"client_params": flh_client_params, "server_params": flh_server_params}

    experiment_list = []

    data_list = []
    for i in range(0, 10):
        data_list.append(generate_zipf_data(d=2 ** (i + 2), n=100000))

    k_list = [10000]
    norms = ["None", "no norm", "norm", "prob simplex", "threshold cut"]
    for i, data in enumerate(data_list):
        for l, norm_type in enumerate(norms):
            for j in range(0, repeats):
                flh_params = copy.deepcopy(FLH)
                freq_oracles = [flh_params]

                for fo in freq_oracles:
                    fo["data"] = data
                    fo["server_params"]["normalization"] = l - 1 if norm_type != "None" else "None"

                for k in k_list:
                    flh_params["client_params"]["k"] = k
                    flh_params["server_params"]["k"] = k
                    experiment_list.append(
                        (("FastLH", "d =" + str(2 ** (i + 2)) + " norm_type=" + str(norm_type)),
                         copy.deepcopy(flh_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# Figure 11
def group6_normalise_fo_high_d():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    repeats = 5

    blh_client_params = {"epsilon": 3, "use_olh": False, "g": 2}
    blh_server_params = {"epsilon": 3, "use_olh": False, "g": 2}

    olh_client_params = copy.deepcopy(blh_client_params)
    olh_server_params = copy.deepcopy(blh_server_params)
    olh_client_params["use_olh"] = True
    olh_server_params["use_olh"] = True

    flh_client_params = copy.deepcopy(olh_client_params)
    flh_server_params = copy.deepcopy(olh_server_params)
    FLH = {"client_params": flh_client_params, "server_params": flh_server_params}

    experiment_list = []

    d_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    data_list = []
    for d in d_list:
        data_list.append(generate_zipf_data(d=d, n=1000000, large_domain=True))

    k_list = [500]
    norms = ["None", "no norm", "norm", "prob simplex", "threshold cut"]
    for i, data in enumerate(data_list):
        for l, norm_type in enumerate(norms):
            for j in range(0, repeats):
                flh_params = copy.deepcopy(FLH)
                freq_oracles = [flh_params]

                for fo in freq_oracles:
                    fo["data"] = data
                    fo["server_params"]["normalization"] = l - 1 if norm_type != "None" else "None"

                for k in k_list:
                    flh_params["client_params"]["k"] = k
                    flh_params["server_params"]["k"] = k
                    experiment_list.append(
                        (("FastLH", "d =" + str(d_list[i]) + " norm_type=" + str(norm_type)),
                         copy.deepcopy(flh_params)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)


def group6_normalise_sketch():
    top_k = 50
    simulation = FrequencyOracleSimulation([0], "", display_full_stats=True, calc_top_k=top_k, autosave=True)
    k = 32
    m = 1024
    repeats = 5
    e = 3
    experiment_list = []

    d_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    data_list = []
    for d in d_list:
        data_list.append(generate_zipf_data(d=d, n=1000000, large_domain=True))

    sketch_types = [False]
    norms = ["None", "no norm", "norm", "prob simplex", "threshold cut"]
    for _ in range(0, repeats):
        for i, data in enumerate(data_list):
            for j, internal_norm in enumerate(norms):
                for l, norm_type in enumerate(norms):
                    for sketch_type in sketch_types:
                        sr = {"server_params": {}, "client_params": {}}
                        sr["server_params"]["m"] = m
                        sr["client_params"]["m"] = m
                        sr["server_params"]["k"] = k
                        sr["client_params"]["k"] = k
                        sr["client_params"]["epsilon"] = e
                        sr["server_params"]["epsilon"] = e
                        sr["server_params"]["sketch_method"] = 1
                        sr["server_params"]["count_sketch"] = sketch_type
                        sr["client_params"]["count_sketch"] = sketch_type
                        sr["server_params"]["normalization"] = l - 1 if norm_type != "None" else "None"
                        sr["server_params"]["estimator_norm"] = j - 1 if internal_norm != "None" else "None"
                        sr["data"] = data

                        # SR with FLH
                        SR_flh = copy.deepcopy(sr)
                        SR_flh["server_params"]["lh_k"] = 500
                        SR_flh["client_params"]["lh_k"] = 500

                        if sketch_type:
                            name = " CS "
                        else:
                            name = " "
                        experiment_list.append(
                            (("SketchResponse",
                              "SR FLH Median" + name + "internal_norm=" + internal_norm + " estimator_norm=" + norm_type,
                              "d=" + str(d_list[i])), copy.deepcopy(SR_flh)))

    simulation.run_and_plot(experiment_list, display_stats_only=True)

# ----------------------------
# HEAVY HITTER EXPERIMENTS
# ----------------------------

def clean_aol():
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = pd.read_csv("/Users/samuelmaddock/Documents/GitHub/Apple-Differential-Privacy/simulations/data/aol.txt",
                     index_col=0, delimiter="\t", names=["AnonID", "Query", "QueryTime", "ItemRank", "ClickURL"])
    df = df.dropna(subset=["ClickURL"], axis="rows")
    df["ClickURL"] = df["ClickURL"].str.replace("http://", "")
    df["ClickURL"] = df["ClickURL"].str.replace("https://", "")
    df["ClickURL"] = df["ClickURL"].str.replace("www.", "")
    print("DataFrame Loaded...")

    counter = Counter(df["ClickURL"].values)
    print(counter.most_common(30))
    print("Size of dataset:", sum(counter.values()))
    print("Unique no of strings", len(counter.keys()))
    df.to_csv("./data/urls.csv", columns=["ClickURL"])
    print("Dataset Saved")

def map_aol_to_bin():
    df = pd.read_csv("./data/urls.csv")
    print("Mapping URLs to binary...")

    df["ClickURL"] = df["ClickURL"].astype(str)
    df["ClickURL"] = df["ClickURL"].map(lambda x: "".join(format(ord(i), 'b').zfill(8) for i in x))
    counter = Counter(df["ClickURL"].values)
    print(counter.most_common(30))
    df.to_csv("./data/bin_urls.csv", columns=["ClickURL"])
    print("Dataset Saved")

def hh_aol():
    # data = pd.read_csv("./data/bin_urls.csv")
    data = pd.read_csv("./data/urls.csv")
    data = data.dropna()
    counter = Counter(data["ClickURL"].values)
    print(counter.most_common(16))

    word_sample_size = data["ClickURL"].nunique()
    word_length = 6
    fragment_length = 2
    start_length = 2

    # alphabet = ["0", "1"]
    alphabet = list(string.ascii_lowercase)
    alphabet.append(".")

    hh_sim = HeavyHitterExperiment(word_length, word_sample_size, autosave=True, data=data["ClickURL"].values,
                                   alphabet=alphabet, metric_k=10)

    # Sketch Parameters - k,m
    m = 1024
    k = 128
    epsilon = 3

    # FLH params
    olh_client_params = {"epsilon": epsilon, "use_olh": True, "g": 2}
    olh_server_params = {"epsilon": epsilon, "use_olh": True, "g": 2}
    flh_client_params = copy.deepcopy(olh_client_params)
    flh_server_params = copy.deepcopy(olh_server_params)
    flh_client_params["k"] = 500
    flh_server_params["k"] = 500
    fast_olh = {"server_params": flh_server_params, "client_params": flh_client_params}

    # HR params
    hr_client_params = {"epsilon": epsilon}
    hr_server_params = {"epsilon": epsilon}
    hr = {"client_params": hr_client_params, "server_params": hr_server_params}

    # SR FLH params
    sketch = copy.deepcopy(fast_olh)
    sketch["client_params"]["m"] = m
    sketch["client_params"]["k"] = k
    sketch["server_params"]["k"] = k
    sketch["server_params"]["m"] = m
    sketch["client_params"]["count_sketch"] = False
    sketch["server_params"]["count_sketch"] = False
    sketch["server_params"]["lh_k"] = fast_olh["server_params"]["k"]
    sketch["client_params"]["lh_k"] = fast_olh["client_params"]["k"]
    sketch["server_params"]["sketch_method"] = 1

    # SR HR params
    sketch_hr = copy.deepcopy(sketch)
    sketch_hr["client_params"]["count_sketch"] = False
    sketch_hr["server_params"]["count_sketch"] = False
    sketch_hr["server_params"]["fo_server"] = HadamardResponseServer(**hr["server_params"], d=m)
    sketch_hr["client_params"]["fo_client"] = HadamardResponseClient(**hr["client_params"], d=m,
                                                                     hash_funcs=sketch_hr["server_params"][
                                                                         "fo_server"].get_hash_funcs())
    sketch_hr["server_params"]["sketch_method"] = 1

    # CMS Params
    cms_params = {"m": m, "k": k, "epsilon": epsilon}
    cms = {"client_params": cms_params, "server_params": cms_params}

    # HCMS Params
    hcms_params = {"m": m, "k": k, "epsilon": epsilon}
    hcms = {"client_params": hcms_params, "server_params": hcms_params}
    hcms["client_params"]["is_hadamard"] = True
    hcms["server_params"]["is_hadamard"] = True

    hh = {"freq_oracle": "cms", "freq_oracle_params": cms,
          "hh_client_params": {"epsilon": epsilon, "max_string_length": word_length, "fragment_length": fragment_length,
                               "start_length": start_length}}

    hh["hh_server_params"] = hh["hh_client_params"]
    hh["k"] = 10

    # CMS
    hh1 = copy.deepcopy(hh)
    hh1["memory_safe"] = True
    hh1["alt_fo_name"] = "CMS"

    # HCMS
    hh2 = copy.deepcopy(hh)
    hh2["freq_oracle"] = "cms"
    hh2["freq_oracle_params"] = hcms
    hh2["alt_fo_name"] = "HCMS"

    # Sketch FLH
    hh3 = copy.deepcopy(hh)
    hh3["freq_oracle"] = "SketchResponse"
    hh3["freq_oracle_params"] = sketch
    hh3["alt_fo_name"] = "CM FLH"

    # Sketch HR
    hh4 = copy.deepcopy(hh)
    hh4["freq_oracle"] = "SketchResponse"
    hh4["freq_oracle_params"] = sketch_hr
    hh4["alt_fo_name"] = "CM HR"

    experiment_list = []

    sketch_sizes = [(128, 1024), (1024, 2048)]
    sketch_sizes = [(32, 1024)]
    hh_list = [hh1, hh2, hh3]
    hh_k_list = [10,20]
    sketch_types = [1]
    sketch_names = ["Median"]

    for item in sketch_sizes:
        for hh_k in hh_k_list:
            for hh in hh_list:
                hh_copy = copy.deepcopy(hh)
                hh["k"] = hh_k
                hh_copy["freq_oracle_params"]["client_params"]["k"] = item[0]
                hh_copy["freq_oracle_params"]["client_params"]["m"] = item[1]
                hh_copy["freq_oracle_params"]["server_params"]["k"] = item[0]
                hh_copy["freq_oracle_params"]["server_params"]["m"] = item[1]

                if hh_copy["alt_fo_name"] == "CM HR":
                    hh_copy["freq_oracle_params"]["server_params"]["fo_server"] = HadamardResponseServer(**hr["server_params"], d=item[1])
                    hh_copy["freq_oracle_params"]["client_params"]["fo_client"] = HadamardResponseClient(**hr["client_params"], d=item[1],
                                                                                     hash_funcs= hh_copy["freq_oracle_params"]["server_params"]["fo_server"].get_hash_funcs())

                hh_copy["alt_fo_name"] = hh_copy["alt_fo_name"] + " T=" + str(hh_k) + " k=" + str(item[0]) + " m=" + str(item[1])

                try:
                    name = copy.deepcopy(hh_copy["alt_fo_name"])
                    for i in sketch_types:
                        a = hh_copy["freq_oracle_params"]["server_params"]["sketch_method"]
                        hh_copy["freq_oracle_params"]["server_params"]["sketch_method"] = i
                        hh_copy["alt_fo_name"] = name + " " + sketch_names[i-1]
                        for i in range(0, 5):
                            experiment_list.extend([("PEM", copy.deepcopy(hh_copy)), ("SFP", copy.deepcopy(hh_copy)), ("TreeHist", copy.deepcopy(hh_copy))])

                except KeyError:
                    for i in range(0,5):
                        experiment_list.extend([("PEM", copy.deepcopy(hh_copy)), ("SFP", copy.deepcopy(hh_copy)), ("TreeHist", copy.deepcopy(hh_copy))])

    names = []
    for item in experiment_list:
        names.append((item[0], item[1]["alt_fo_name"]))

    for name in set(names):
        print(name)
    print("Total Number of Experiments", len(experiment_list))
    print(len(set(names)), "unique experiments")
    print("\n")
    hh_sim.run_and_plot(experiment_list, display_stats_only=True)

# ----------------------------
## GENERATING DATA
# ----------------------------

# generate_zipf_data(1024, name="zipf_1024")

# ----------------------------
## RUNNING EXPERIMENTS
# ----------------------------


# ----------------------------
# GROUP 1 EXPERIMENTS
# ----------------------------

# group1_vary_eps()
# group1_vary_d()
# group1_vary_n()

# ----------------------------
# GROUP 2 EXPERIMENTS
# ----------------------------

# group2_vary_eps()
# group2_vary_d()

# ----------------------------
# GROUP 3 EXPERIMENTS
# ----------------------------

# group3_vary_eps()
# group3_vary_d()
# group3_vary_t()
# group3_high_privacy()

# ----------------------------
# GROUP 5 EXPERIMENTS
# ----------------------------

# group5_bloom_reg()

# group5_bloom_comparison()

# group5_SR_vary_k()
# group5_SR_vary_m()

group5_ALL_vary_eps()
# group5_ALL_vary_d()

# ----------------------------
# GROUP 6 EXPERIMENTS
# ----------------------------

# group6_normalise_sketch()

# group6_normalise_fo_high_d()

# group6_normalise_fo_low_d()

# --------------------------------------- HEAVY HITTER EXPERIMENTS ---------------------------------------

#hh_aol()

