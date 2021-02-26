import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
import math

sns.set_theme()
sns.set_style("whitegrid", {'grid.linestyle': '--', "xtick.major.size": 8, "ytick.major.size": 8})
# plt.rc("font", size=12)


plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=15)  # fontsize of the axes title
plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
plt.rc('legend', fontsize=15)  # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title

metric_map = {"mse": "Mean Square Error (MSE)", "k_mse": "Top-k MSE (k=50)"}

# -----------------------------
# GROUP 1 PLOTS
# -----------------------------

def plot_group1_vary_eps():
    df = pd.read_csv("./metrics/group1-varyEps.csv")
    freq_oracles = set(df["freq_oracle"].values)

    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = [".", "s", "^"]

    for i, freq_oracle in enumerate(freq_oracles):
        order = 1
        if "OUE" in freq_oracle:
            order = 2
            marker = "."
            color = "orange"
        elif "DE" in freq_oracle:
            marker = "^"
            color="blue"
        else:
            marker = "s"
            color = "green"

        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        ax.plot(x, means, marker=marker, color=color, label=freq_oracle)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Square Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('group1_vary_eps.eps', format='eps')
    plt.clf()


def plot_group1_vary_d():
    df = pd.read_csv("./metrics/group1-varyD.csv")
    freq_oracles = set(df["freq_oracle"].values)

    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = [".", "s", "^"]

    for i, freq_oracle in enumerate(freq_oracles):
        order  = 1
        if "OUE" in freq_oracle:
            order = 2
            marker = "."
            color = "orange"
        elif "DE" in freq_oracle:
            marker = "^"
            color="blue"
        else:
            marker = "s"
            color = "green"

        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        ax.plot(x.astype("str"), means, marker=marker, color=color, zorder=order, label=freq_oracle)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('group1_vary_d.eps', format='eps')
    plt.clf()

def plot_group1_time():
    df = pd.read_csv("./metrics/group1-varyD.csv")
    freq_oracles = set(df["freq_oracle"].values)

    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = [".", "s", "^"]

    for i, freq_oracle in enumerate(freq_oracles):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["total_time"]
        order  = 1
        if "OUE" in freq_oracle:
            order = 2
            marker = "."
            color = "orange"
        elif "DE" in freq_oracle:
            marker = "^"
            color="blue"
        else:
            marker = "s"
            color = "green"

        ax.plot(x.astype("str"), means, marker=marker, color=color, label=freq_oracle, zorder=order)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Total Time (seconds)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('group1_time.eps', format='eps')
    plt.clf()

def plot_group1_vary_n():
    df = pd.read_csv("./metrics/group1-varyN.csv")
    freq_oracles = set(df["freq_oracle"].values)

    x = [10000, 50000, 100000, 500000, 5000000, 10000000]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = [".", "s", "^"]

    for i, freq_oracle in enumerate(freq_oracles):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        ax.plot(x, means, marker=markers[i], label=freq_oracle)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.show()
    plt.clf()


# -----------------------------
# GROUP 2 PLOTS
# -----------------------------

def plot_group2_vary_eps():
    df = pd.read_csv("./metrics/group2-varyEps.csv")
    df = pd.read_csv("./metrics/BLH Fix 2.csv")
    freq_oracles = set(df["freq_oracle"].values)
    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["^", "s", "^", "*", "d", "o", ">"]
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 5).as_hex()
    flh_count = 0

    for i, freq_oracle in enumerate(freq_oracles):
        if freq_oracle == "FastLH":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["k"] = df["info"].str.split(" ", expand=True)[2]
            for k in filtered_df["k"].unique():
                means = filtered_df[filtered_df["k"] == k].groupby("info", sort=False)["mse"].mean()
                ax.plot(x, means,  marker=markers[0], label="FLH" + " (" + k.replace("k=", "$k^\prime$=") + ")",
                        color=color_dict[0][flh_count])
                flh_count += 1
                # mfc="black", mec="black"
        else:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()["mse"]
            if freq_oracle == "OLH":
                color = "navy"
                marker = "s"
            else:
                color = "forestgreen"
                marker = "."
            ax.plot(x, means, marker=marker, label=freq_oracle, color=color)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig('group2_vary_eps.eps', format='eps')
    plt.clf()

def plot_group2_vary_d():
    df = pd.read_csv("./metrics/group2-varyD.csv")
    df = pd.read_csv("./metrics/BLH Fix 2.csv")

    freq_oracles = set(df["freq_oracle"].values)
    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["^", "s", "^", "*", "d", "o", ">"]
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 5).as_hex()
    flh_count = 0

    for i, freq_oracle in enumerate(freq_oracles):
        if freq_oracle == "FastLH":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["k"] = df["info"].str.split(" ", expand=True)[2]
            for k in filtered_df["k"].unique():
                means = filtered_df[filtered_df["k"] == k].groupby("info", sort=False)["mse"].mean()
                ax.plot(x.astype("str"), means, marker=markers[0],
                        label="FLH" + " (" + k.replace("k=", "$k^\prime$=") + ")", color=color_dict[0][flh_count])
                flh_count += 1
        else:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()["mse"]
            if freq_oracle == "OLH":
                color = "navy"
                marker = "s"
            else:
                color = "forestgreen"
                marker = "."

            ax.plot(x.astype("str"), means, marker=marker, label=freq_oracle, color=color)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig('group2_vary_d.eps', format='eps')
    plt.clf()

def plot_group2_time():
    df = pd.read_csv("./metrics/group2-varyD.csv")
    freq_oracles = set(df["freq_oracle"].values)
    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["^", "s", "^", "*", "d", "o", ">"]

    metric = "total_time"
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 5).as_hex()
    flh_count = 0

    for i, freq_oracle in enumerate(freq_oracles):
        if freq_oracle == "FastLH":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["k"] = df["info"].str.split(" ", expand=True)[2]
            for k in filtered_df["k"].unique():
                means = filtered_df[filtered_df["k"] == k].groupby("info", sort=False)[metric].mean()
                ax.plot(x.astype("str"), means, marker=markers[0],
                        label="FLH" + " (" + k.replace("k=", "$k^\prime$=") + ")", color=color_dict[0][flh_count])
                flh_count += 1
        else:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            if freq_oracle == "OLH":
                color = "navy"
                marker = "s"
                order = 1
            else:
                color = "forestgreen"
                marker = "."
                order =2
            ax.plot(x.astype("str"), means, marker=marker, label=freq_oracle, color=color , zorder=order)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Total Time (seconds)")
    ax.set_yscale('log')
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig('group2_time.eps', format='eps')
    plt.clf()

# d=500, n= 1 million
def plot_group2_k():
    plt.figure()
    df = pd.read_csv("./metrics/group2-varyK.csv")
    filtered_df = df.groupby("freq_oracle", sort=False).mean()["mse"]
    olh_err = filtered_df["freq_oracle" == "optimised_local_hashing"]
    avg_errs = filtered_df.iloc[1:].values
    approx_ratios = avg_errs / olh_err

    N = 1000000
    inc = 100
    iters = 200

    ax = sns.lineplot(x=range(100, inc * (iters + 1), inc), y=avg_errs, label="FLH")
    plt.axhline(olh_err, color="r", label="OLH", linewidth=0.8)
    ax.set_yscale('log')

    plt.xlabel("Number of Hash Functions ($k^\prime$)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("group2_k.eps", format='eps')
    plt.clf()

# -----------------------------
# GROUP 3 PLOTS
# -----------------------------

def plot_group3_vary_T():
    df = pd.read_csv("./metrics/group3-varyT.csv")
    df["e"] = df["info"].str.split(" ", expand=True)[0]
    df["t"] = df["info"].str.split(" ", expand=True)[1]

    x = [1, 2, 3, 4, 5, 6, 7]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = [".", "s", "^", "*", "d", "o", ">"]
    colors = sns.color_palette("crest", 6).as_hex()

    for i, freq_oracle in enumerate(df["e"].unique()):
        filtered_df = df[df["e"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        upper = filtered_df.groupby("info", sort=False).max()["mse"] - means
        lower = means - filtered_df.groupby("info", sort=False).min()["mse"]
        errs = [upper, lower]
        ax.plot(x, means, color=colors[i], marker=markers[2], label=freq_oracle.replace("e", "$\epsilon$"))
        # ax.errorbar(x, means, yerr=errs, capsize=20, marker=markers[i], label=freq_oracle)

    # ax = sns.barplot(x="t", y="mse", hue="e", data=df)
    plt.xlabel("Number of Hadamard Coefficients Sampled ($t$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig('group3_vary_t.eps', format='eps')


def plot_group3_vary_eps():
    df = pd.read_csv("./metrics/group3-varyEps2.csv")

    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">"]
    t_colors = sns.color_palette("Greens_r", n_colors=20)

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if freq_oracle == "HadamardMech":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["t"] = filtered_df["info"].str.split(" ", expand=True)[1]
            filtered_df["e"] = filtered_df["info"].str.split(" ", expand=True)[0]
            for j, t in enumerate(filtered_df["t"].unique()):
                means = filtered_df[filtered_df["t"] == t].groupby("e", sort=False).mean()["mse"]

                ax.plot(x, means, marker=markers[1], label="HM" + " " + t, color=t_colors[(j*3) + 1])

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if freq_oracle == "HR":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()["mse"]
            ax.plot(x, means, marker=markers[3], label=freq_oracle, color="purple")

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.savefig('group3_vary_eps.eps', format='eps')
    plt.clf()


def plot_group3_vary_d():
    df = pd.read_csv("./metrics/group3-varyD3.csv")
    df["e"] = df["info"].str.split(" ", expand=True)[0]

    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">"]
    t_colors = sns.color_palette("Greens_r", n_colors=20)

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if freq_oracle == "HadamardMech":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["t"] = filtered_df["info"].str.split(" ", expand=True)[1]
            filtered_df["d"] = filtered_df["info"].str.split(" ", expand=True)[0]
            for j, t in enumerate(filtered_df["t"].unique()):
                means = filtered_df[filtered_df["t"] == t].groupby("d", sort=False).mean()["mse"]
                ax.plot(x.astype("str"), means, marker=markers[1], label="HM" + " " + t, color=t_colors[(j*3) + 1])

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if freq_oracle == "HR":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()["mse"]
            ax.plot(x.astype("str"), means, marker=markers[3], label=freq_oracle, color="purple")

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig('group3_vary_d.eps', format='eps')
    plt.clf()


def plot_group3_lowEps():
    df = pd.read_csv("./metrics/group3-lowEps10.csv")
    x = np.arange(0.1, 1.1, 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["*", ".", "^", "*", "d", "o", ">"]
    colors = ["purple", "green"]
    trans1 = matplotlib.transforms.Affine2D().translate(-0.0, 0.0) + ax.transData
    trans2 = matplotlib.transforms.Affine2D().translate(+0.0, 0.0) + ax.transData
    trans = [trans1, trans2]
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        upper = filtered_df.groupby("info", sort=False).max()["mse"] - means
        lower = means - filtered_df.groupby("info", sort=False).min()["mse"]
        errs = [upper, lower]

        name = freq_oracle
        if name == "HadamardMech":
            name = "HM"

        ax.errorbar(x, means, yerr=errs, capsize=5, marker=markers[i], color=colors[i], label=name, transform=trans[i])
        # ax.plot(x, means, marker=markers[i], label=freq_oracle)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('group3_lowEps.eps', format='eps')


# -----------------------------
# GROUP 4 PLOTS
# -----------------------------

def plot_group4_vary_eps():
    df = pd.read_csv("./metrics/group4-varyEps.csv")
    x = np.arange(0.5, 5.5, 0.5)
    t = [1, 1, 2, 3, 4, 5]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    color_dict = {
        "OLH": "navy",
        "OUE": "coral",
        "FastLH": "skyblue",
        "HadamardMech": "green",
        "HR": "purple"
    }
    ignore_list = ["DE", "BLH", "SUE"]
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        color = color_dict.get(freq_oracle, "blue")
        if freq_oracle == "FastLH":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["k"] = df["info"].str.split(" ", expand=True)[2]
            means = filtered_df[filtered_df["k"] == "k=10000"].groupby("info", sort=False)["mse"].mean()
            ax.plot(x, means, marker=markers[1], color=color, label="FLH ($k^\prime$=10,000)")
        elif freq_oracle == "HadamardMech":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["t"] = filtered_df["info"].str.split(" ", expand=True)[1]
            filtered_df["e"] = filtered_df["info"].str.split(" ", expand=True)[0]
            all_means = []
            for t in [1, 2, 3, 4, 5]:
                all_means.append(filtered_df[filtered_df["t"] == "t=" + str(t)].groupby("e", sort=False).mean()["mse"])
            y = [all_means[0][0], all_means[0][1], all_means[0][2], all_means[1][3], all_means[1][4], all_means[2][5],
                 all_means[2][6], all_means[3][7], all_means[3][8], all_means[4][9]]
            ax.plot(x, y, marker=markers[2], color=color, label="HM")
        elif freq_oracle not in ignore_list:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            name = freq_oracle
            marker = markers[i]
            if name == "OLH":
                marker = "s"
            if freq_oracle == "HR":
                marker = "*"
            means = filtered_df.groupby("info", sort=False).mean()["mse"]
            ax.plot(x, means, marker=marker, color=color, label=freq_oracle)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.savefig('group4_vary_eps.eps', format='eps')


def plot_group4_vary_d():
    df = pd.read_csv("./metrics/group4-varyD.csv")
    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    color_dict = {
        "OLH": "navy",
        "OUE": "coral",
        "FastLH": "skyblue",
        "HadamardMech": "green",
        "HR": "purple"
    }
    ignore_list = ["DE", "BLH", "SUE"]
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        color = color_dict.get(freq_oracle, "blue")

        if freq_oracle == "FastLH":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["k"] = df["info"].str.split(" ", expand=True)[2]
            means = filtered_df[filtered_df["k"] == "k=10000"].groupby("info", sort=False)["mse"].mean()
            ax.plot(x.astype("str"), means, marker=markers[1], color=color, label="FLH ($k^\prime$=10,000)")
        elif freq_oracle == "HadamardMech":
            filtered_df = pd.read_csv("./metrics/group3-varyD3.csv")
            filtered_df = filtered_df[filtered_df["freq_oracle"] == freq_oracle]
            filtered_df["t"] = filtered_df["info"].str.split(" ", expand=True)[1]
            filtered_df["e"] = filtered_df["info"].str.split(" ", expand=True)[0]
            means = filtered_df[filtered_df["t"] == "t=3"].groupby("e", sort=False).mean()["mse"]
            ax.plot(x.astype("str"), means, marker=markers[2], color=color, label="HM" + " (t=3)")
        elif freq_oracle not in ignore_list:
            if freq_oracle == "HR":
                filtered_df = pd.read_csv("./metrics/group3-varyD3.csv")
                filtered_df = filtered_df[filtered_df["freq_oracle"] == freq_oracle]
            else:
                filtered_df = df[df["freq_oracle"] == freq_oracle]

            name = freq_oracle
            marker = markers[i]
            if name == "OLH":
                marker = "s"
            if freq_oracle == "HR":
                marker = "*"
            means = filtered_df.groupby("info", sort=False).mean()["mse"]
            ax.plot(x.astype("str"), means, marker=marker, color=color, label=freq_oracle)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.savefig('group4_vary_d.eps', format='eps')


def plot_group4_time():
    df = pd.read_csv("./metrics/group4-varyD.csv")
    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    ignore_list = ["DE", "BLH", "SUE"]
    metric = "total_time"
    color_dict = {
        "OLH": "navy",
        "OUE": "coral",
        "FastLH": "skyblue",
        "HadamardMech": "green",
        "HR": "purple"
    }
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        color = color_dict.get(freq_oracle, "blue")

        if freq_oracle == "FastLH":
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["k"] = df["info"].str.split(" ", expand=True)[2]
            means = filtered_df[filtered_df["k"] == "k=10000"].groupby("info", sort=False)[metric].mean()
            ax.plot(x.astype("str"), means, marker=markers[1], color=color, label="FLH ($k^\prime$=10,000)")
        elif freq_oracle == "HadamardMech":
            filtered_df = pd.read_csv("./metrics/group3-varyD3.csv")
            filtered_df = filtered_df[filtered_df["freq_oracle"] == freq_oracle]
            filtered_df["t"] = filtered_df["info"].str.split(" ", expand=True)[1]
            filtered_df["e"] = filtered_df["info"].str.split(" ", expand=True)[0]
            means = filtered_df[filtered_df["t"] == "t=3"].groupby("e", sort=False).mean()[metric]
            ax.plot(x.astype("str"), means, marker=markers[2], color=color, label="HM" + " (t=3)")
        elif freq_oracle not in ignore_list:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            name = freq_oracle
            marker = markers[i]
            if name == "OLH":
                marker = "s"
            if freq_oracle == "HR":
                marker = "*"
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            ax.plot(x.astype("str"), means, marker=marker, color=color, label=freq_oracle)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Total Time (seconds)")
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.savefig('group4_vary_time.eps', format='eps')


# -----------------------------
#  GROUP 5 PLOTS
# -----------------------------

def plot_group5_vary_k():
    df = pd.read_csv("./metrics/group5-varyK.csv")
    x = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    metric = "mse"
    epsilons = ["e=0.5", "e=1", "e=3"]
    color_dict = {}
    color_dict[0] = sns.color_palette("dark:b", 5).as_hex()
    color_dict[1] = sns.color_palette("dark:orange", 5).as_hex()

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        for j, eps in enumerate(epsilons):
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["e"] = df["info"].str.split(" ", expand=True)[1]
            filtered_df = filtered_df[filtered_df["e"] == eps]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            eps = eps.replace("e", "$\epsilon$")
            name = freq_oracle.upper()
            ax.plot(x.astype("str"), means, color=color_dict[i][j + 2], marker=markers[i],
                    label=name + " (" + eps + ")")

    ax.set_yscale('log')
    plt.xlabel("Number of Hash Functions ($r$)")
    plt.ylabel("Mean Square Error (MSE)")
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_vary_k.eps', format='eps')


def plot_group5_vary_m():
    df = pd.read_csv("./metrics/group5-varyMFinal.csv")
    x = np.array([32, 64, 128, 256, 512, 1024, 2048])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    metric = "mse"
    epsilons = ["e=0.5", "e=1", "e=3"]

    color_dict = {}
    color_dict[0] = sns.color_palette("dark:b", 5).as_hex()
    color_dict[1] = sns.color_palette("dark:orange", 5).as_hex()

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        for j, eps in enumerate(epsilons):
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["e"] = df["info"].str.split(" ", expand=True)[1]
            filtered_df = filtered_df[filtered_df["e"] == eps]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            eps = eps.replace("e", "$\epsilon$")
            name = freq_oracle.upper()
            ax.plot(x.astype("str"), means, color=color_dict[i][j + 2], marker=markers[i],
                    label=name + " (" + eps + ")")

    ax.set_yscale('log')
    plt.xlabel("Size of Sketch Vector ($c$)")
    plt.ylabel("Mean Square Error (MSE)")
    plt.legend(fontsize=13.5)
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_vary_m.eps', format='eps')


def plot_group5_SR_vary_k(metric="mse", same_plot=True):
    df = pd.read_csv("./metrics/group5-SR-varyK.csv")
    x = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    fig = plt.figure(figsize=(12,5))
    #ax = fig.add_subplot(111)

    markers = [".", "^", "*", "d", "o", ">", ".", "."]
    markers = ["o", "o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Greens_r", 5).as_hex()
    color_dict[1] = sns.color_palette("Purples_r", 5).as_hex()
    color_dict[2] = sns.color_palette("Reds_r", 5).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            color = color_dict[int(i / 2)][i % 2]
            name = freq_oracle.replace("SR", "CM").replace("CM CS", "CS")
            marker = "o"
            if "CS" in name:
                marker = "s"
            elif "CM" in name:
                marker = "^"
            ax.plot(x.astype("str"), means, marker=marker, label=name, color=color)

        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel("Number of Hash Functions ($r$)", fontsize=12)
        ax.set_ylabel(metric_map.get(metric), fontsize=12)

    #plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(-0.5, 1.5))
    #plt.legend(loc='center center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=11)
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.2), fontsize=11, ncol=2)
    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if same_plot:
        plt.savefig('group5_SR_vary_k.eps', format='eps')
    else:
        plt.savefig('group5_SR_vary_k_' + metric + '.eps', format='eps')


def plot_group5_SR_vary_m(metric="mse", same_plot=True):
    # df = pd.read_csv("./metrics/group5-SR-varyM.csv")
    df = pd.read_csv("./metrics/group5-SR-varyM FINAL.csv")

    x = np.array([32, 64, 128, 256, 512, 1024, 2048])
    fig = plt.figure(figsize=(12,5))
    markers = ["o", "o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Greens_r", 5).as_hex()
    color_dict[1] = sns.color_palette("Purples_r", 5).as_hex()
    color_dict[2] = sns.color_palette("Reds_r", 5).as_hex()
    color_dict[3] = sns.color_palette("Oranges_r", 5).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "Mean" in freq_oracle and "Debias Mean" not  in freq_oracle:
                print("skip")
            else:
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                color = color_dict[int(i / 2)][i % 2]
                name = freq_oracle.replace("SR", "CM").replace("CM CS", "CS")
                marker = "o"
                if "CS" in name:
                    marker = "s"
                elif "CM" in name:
                    marker = "^"
                ax.plot(x.astype("str"), means, marker=marker, label=name, color=color)

        ax.set_yscale('log')
        ax.set_xlabel("Size of Sketch Vector ($c$)", fontsize=12)
        ax.set_ylabel(metric_map.get(metric), fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.plot()
    if same_plot:
        plt.savefig('group5_SR_vary_m.eps', format='eps')
    else:
        plt.savefig('group5_SR_vary_m_' + metric + '.eps', format='eps')

def plot_group5_SR_vary_k_individual(metric="mse"):
    # df = pd.read_csv("./metrics/group5-SR-varyK.csv")
    df = pd.read_csv("./metrics/group5-SR-varyK FINAL.csv")

    x = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    fig = plt.figure(figsize=(8,5))

    markers = [".", "^", "*", "d", "o", ">", ".", "."]
    markers = ["o", "o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Greens_r", 5).as_hex()
    color_dict[1] = sns.color_palette("Purples_r", 5).as_hex()
    color_dict[2] = sns.color_palette("Reds_r", 5).as_hex()
    color_dict[3] = sns.color_palette("Reds_r", 5).as_hex()

    ax = fig.add_subplot(111)

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if "Mean" in freq_oracle and "Debias Mean" not  in freq_oracle:
            print("skip")
        else:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            color = color_dict[int(i / 2)][i % 2]
            name = freq_oracle.replace("SR", "CM").replace("CM CS", "CS").replace("Debias Mean", "Mean")
            marker = "o"
            if "CS" in name:
                marker = "s"
            elif "CM" in name:
                marker = "^"
            ax.plot(x.astype("str"), means, marker=marker, label=name, color=color)

    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel("Number of Hash Functions ($r$)", fontsize=12)
    ax.set_ylabel(metric_map.get(metric), fontsize=12)

    #plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(-0.5, 1.5))
    #plt.legend(loc='center center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=11)
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.2), fontsize=11, ncol=2)
    plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('group5_SR_vary_k_' + metric + '.eps', format='eps')

def plot_group5_SR_vary_m_individual(metric="mse"):
    # df = pd.read_csv("./metrics/group5-SR-varyM.csv")
    df = pd.read_csv("./metrics/group5-SR-varyM FINAL.csv")

    x = np.array([32, 64, 128, 256, 512, 1024, 2048])
    fig = plt.figure(figsize=(6,5))
    markers = ["o", "o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Greens_r", 5).as_hex()
    color_dict[1] = sns.color_palette("Purples_r", 5).as_hex()
    color_dict[2] = sns.color_palette("Reds_r", 5).as_hex()
    color_dict[3] = sns.color_palette("Reds_r", 5).as_hex()

    ax = fig.add_subplot(111)

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if "Mean" in freq_oracle and "Debias Mean" not in freq_oracle:
            print("skip")
        else:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            color = color_dict[int(i / 2)][i % 2]
            name = freq_oracle.replace("SR", "CM").replace("CM CS", "CS").replace("Debias Mean", "Mean")
            marker = "o"
            if "CS" in name:
                marker = "s"
            elif "CM" in name:
                marker = "^"
            ax.plot(x.astype("str"), means, marker=marker, label=name, color=color)

    ax.set_yscale('log')
    ax.set_xlabel("Size of Sketch Vector ($c$)", fontsize=12)
    ax.set_ylabel(metric_map.get(metric), fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_SR_vary_m_' + metric + '.eps', format='eps')

def plot_group5_bloom_comparison(metric="mse", same_plot=True):
    # df = pd.read_csv("./metrics/group5-bloom-comparisons.csv")
    df = pd.read_csv("./metrics/group5_bloom_comparisons_FINAL.csv")

    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure(figsize=(12,4))
    #ax = fig.add_subplot(111)
    markers = ["o", "o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues_r", 5).as_hex()
    # color_dict[1] = sns.color_palette("Blues_r", 5).as_hex()[2:]
    color_dict[1] = sns.color_palette("Greens_r", 5).as_hex()
    color_dict[2] = sns.color_palette("Purples_r", 5).as_hex()
    color_dict[3] = sns.color_palette("Reds_r", 5).as_hex()
    color_dict[4] = sns.color_palette("Reds_r", 5).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "rappor" in freq_oracle and "reg=0.005" not in freq_oracle:
                print("skip")
            if "Mean" in freq_oracle and "Debias Mean" not in freq_oracle:
                print("skip")
            else:
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                filtered_df["d"] = range(0, len(filtered_df))
                filtered_df["d"] = filtered_df["d"].apply(lambda j: x[int(j % 10)])
                means = filtered_df.groupby("d", sort=False).mean()[metric]
                name = freq_oracle.replace("rappor", "Bloom Filter with DE").replace("SR", "CM").replace("CM CS", "CS").replace("Debias Mean", "Mean")

                color = color_dict[math.ceil(i / 2)][i % 2]
                marker = "o"
                if "CS" in name:
                    marker = "s"
                elif "CM" in name:
                    marker = "^"
                ax.plot(x, means, marker=marker, label=name, color=color)

        ax.set_xlabel("Domain Size ($d$)", fontsize=12)
        ax.set_ylabel(metric_map.get(metric), fontsize=12)
        ax.set_yscale('log')
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    #plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0))
    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.plot()

    if same_plot:
         plt.savefig('group5_bloom_comparisons.eps', format='eps', bbox_inches="tight")
    else:
        plt.savefig('group5_bloom_comparisons_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_rappor(metric="mse"):
    df = pd.read_csv("./metrics/group5-RAPPOR.csv")
    x = np.array([0, 0.0001, 0.0005, 0.00075, 0.001, 0.005, 0.0075, 0.01, 0.05, 0.075, 0.1, 0.5, 0.75, 1, 5])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", "o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]

    m_list = ["32", "64", "128"]
    df["reg"] = df["freq_oracle"].str.split("reg=", expand=True)[1]
    df["m"] = df["info"].str.split("m=", expand=True)[1]

    for m in m_list:
        filtered_df = df[df["m"] == m]
        means = filtered_df.groupby("reg", sort=False).mean()[metric]
        ax.plot(x, means, marker="o", label="RAPPOR m=" + str(m))

    plt.xlabel(r"L2 Regularisation ($\alpha$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.legend()
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_RAPPOR_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_apple_vary_eps():
    df = pd.read_csv("./metrics/group5-apple-varyEps.csv")
    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        ax.plot(x, means, marker=markers[i], label=freq_oracle)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.plot()
    plt.savefig('group5_apple_vary_eps.eps', format='eps')


def plot_group5_apple_vary_d():
    df = pd.read_csv("./metrics/group5-apple-varyD.csv")
    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()["mse"]
        ax.plot(x.astype("str"), means, marker=markers[i], label=freq_oracle)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    plt.legend()
    plt.plot()
    plt.savefig('group5_apple_vary_d.eps', format='eps')


def plot_group5_SR_vary_eps():
    df = pd.read_csv("./metrics/group5-SR-varyEps.csv")
    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    metric = "mse"
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
        ax.plot(x, means, marker=markers[i], label=name)

    plt.xticks(x)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Mean Squared Error (MSE)")
    ax.set_yscale('log')
    ax.legend()
    plt.plot()
    plt.savefig('group5_SR_vary_eps_' + metric + '.eps', format='eps')


def plot_group5_SR_vary_d():
    df = pd.read_csv("./metrics/group5-SR-varyD.csv")
    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    metric = "mse"
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
        ax.plot(x, means, marker=markers[i], label=name)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel("Mean Squared Error (MSE)")
    # ax.set_yscale('log')
    ax.legend()
    plt.plot()
    plt.savefig('group5_SR_vary_d_' + metric + '.eps', format='eps')


def plot_group5_debias(same_plot=True):
    df = pd.read_csv("./metrics/Debias Test.csv")

    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure(figsize=(12,4))
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 12).as_hex()
    color_dict[1] = sns.color_palette("Purples", 12).as_hex()
    color_dict[2] = sns.color_palette("Greens", 12).as_hex()
    color_dict[3] = sns.color_palette("Oranges", 12).as_hex()
    color_dict[4] = sns.color_palette("Reds", 3).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        min_count = 2
        med_count = 2
        mean_count = 2
        debias_count = 2

        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "SR" in freq_oracle and "Min" not in freq_oracle:
                if "Median" in freq_oracle:
                    color = color_dict[0][(med_count+1)*2+3]
                    marker = markers[3]
                    med_count += 1
                elif "Debias" in freq_oracle:
                    color = color_dict[1][(debias_count+1)*2+3]
                    marker = markers[2]
                    debias_count += 1
                elif "Min" in freq_oracle:
                    color = color_dict[2][(min_count+1)*2+3]
                    marker = markers[1]
                    min_count += 1
                elif "Mean" in freq_oracle:
                    color = color_dict[3][(mean_count+1)*2+3]
                    marker = markers[1]
                    mean_count += 1

                filtered_df = df[df["freq_oracle"] == freq_oracle]
                print(filtered_df)
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
                ax.plot(x, means, marker=marker, label=name, color=color)

        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "hcms" in freq_oracle:
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                color = color_dict[4][1]
                ax.plot(x, means, marker="o", label=freq_oracle.upper(), color=color)

        ax.set_xlabel("Domain Size ($d$)")
        ax.set_ylabel(metric_map.get(metric))
        ax.set_yscale('log')
        ax.tick_params(axis="x", labelsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout()
    plt.plot()

    plt.savefig('group5_debias.eps', format='eps')

def plot_group5_all_vary_d(metric="mse"):
    df = pd.read_csv("./metrics/group5-SR-varyD.csv")
    apple_df = pd.read_csv("./metrics/group5-apple-varyD.csv")
    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 3).as_hex()
    color_dict[1] = sns.color_palette("Purples", 3).as_hex()
    color_dict[2] = sns.color_palette("Greens", 3).as_hex()
    color_dict[3] = sns.color_palette("Reds", 2).as_hex()

    hr_count = 0
    hm_count = 0
    flh_count = 0
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if "HR" in freq_oracle:
            color = color_dict[0][hr_count]
            marker = markers[3]
            hr_count += 1
        elif "HM" in freq_oracle:
            color = color_dict[1][hm_count]
            marker = markers[2]
            hm_count += 1
        elif "FLH" in freq_oracle:
            color = color_dict[2][flh_count]
            marker = markers[1]
            flh_count += 1

        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
        ax.plot(x, means, marker=marker, label=name, color=color)

    for i, freq_oracle in enumerate(apple_df["freq_oracle"].unique()):
        filtered_df = apple_df[apple_df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        ax.plot(x, means, marker="o", label=freq_oracle, color=color_dict[3][i])

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_ALL_vary_d_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_all_vary_eps(metric="mse"):
    df = pd.read_csv("./metrics/group5-SR-varyEps.csv")
    apple_df = pd.read_csv("./metrics/group5-apple-varyEps.csv")
    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 3).as_hex()
    color_dict[1] = sns.color_palette("Purples", 3).as_hex()
    color_dict[2] = sns.color_palette("Greens", 3).as_hex()
    color_dict[3] = sns.color_palette("Reds", 2).as_hex()

    hr_count = 0
    hm_count = 0
    flh_count = 0
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        if "HR" in freq_oracle:
            color = color_dict[0][hr_count]
            marker = markers[3]
            hr_count += 1
        elif "HM" in freq_oracle:
            color = color_dict[1][hm_count]
            marker = markers[2]
            hm_count += 1
        elif "FLH" in freq_oracle:
            color = color_dict[2][flh_count]
            marker = markers[1]
            flh_count += 1

        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
        ax.plot(x, means, marker=marker, label=name, color=color)

    for i, freq_oracle in enumerate(apple_df["freq_oracle"].unique()):
        filtered_df = apple_df[apple_df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        ax.plot(x, means, marker="o", label=freq_oracle, color=color_dict[3][i])

    plt.xlabel("Privacy Budget Size ($\epsilon$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    ax.tick_params(axis="x", labelsize=8)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0), fontsize=13)
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_ALL_vary_eps_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_fo_vary_d(metric="mse", same_plot=True):
    # df = pd.read_csv("./metrics/sketch_all_d.csv")
    df = pd.read_csv("./metrics/group5-ALL-varyD FINAL.csv")

    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure(figsize=(12,4))
    markers = ["o", "s", "^", "*", "d", "o", ">", ".", "."]

    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 12).as_hex()
    color_dict[1] = sns.color_palette("Purples", 12).as_hex()
    color_dict[2] = sns.color_palette("Greens", 12).as_hex()
    color_dict[3] = sns.color_palette("Reds", 3).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        hr_count = 2
        hm_count = 2
        flh_count = 2

        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "t=1" in freq_oracle:
                print("skip")
            elif "Median" in freq_oracle:
                if "HR" in freq_oracle:
                    color = color_dict[0][(hr_count+1)*2+3]
                    marker = markers[3]
                    hr_count += 1
                elif "HM" in freq_oracle:
                    color = color_dict[1][(hm_count+1)*2+3]
                    marker = markers[2]
                    hm_count += 1
                elif "FLH" in freq_oracle:
                    color = color_dict[2][(flh_count+1)*2+3]
                    marker = markers[1]
                    flh_count += 1

                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
                ax.plot(x, means, marker=marker, label=name, color=color)

        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "cms" in freq_oracle:
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                color = color_dict[3][0]
                name = "CM (Mean) with OUE"
                if "hcms" in freq_oracle:
                    color = color_dict[3][1]
                    name = "CM (Mean) with HM t=1"
                ax.plot(x, means, marker=".", label=name, color=color)

        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "Debias Mean" in freq_oracle and "t=3" in freq_oracle:
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                color = color_dict[3][2]
                name = "CM (Mean) with HM t=3"
                ax.plot(x, means, marker=".", label=name, color=color)

        ax.set_xlabel("Domain Size ($d$)")
        ax.set_ylabel(metric_map.get(metric))
        ax.set_yscale('log')
        ax.tick_params(axis="x", labelsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout()
    plt.plot()

    if same_plot:
        plt.savefig('group5_fo_vary_d.eps', format='eps')
    else:
        plt.savefig('group5_fo_vary_d_' + metric + '.eps', format='eps')


def plot_group5_fo_vary_eps(metric="mse"):
    # df = pd.read_csv("./metrics/sketch_eps_all.csv")
    df = pd.read_csv("./metrics/group5-ALL-varyEps FINAL.csv")

    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", "s", "^", "*", "d", "o", ">", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 12).as_hex()
    color_dict[1] = sns.color_palette("Purples", 12).as_hex()
    color_dict[2] = sns.color_palette("Greens", 12).as_hex()
    color_dict[3] = sns.color_palette("Reds", 3).as_hex()

    hr_count = 2
    hm_count = 2
    flh_count = 2

    df["freq_oracle"] = df["freq_oracle"].str.replace("t=1", "").str.replace("t=2", "").str.replace("t=3", "").str.replace("t=4", "").str.replace("t=5", "")
    print(df["freq_oracle"].unique())

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        print(freq_oracle)
        if "HM" in freq_oracle and "t=optimal" not in freq_oracle:
            print("skip")
        elif "Median" in freq_oracle:
            if "HR" in freq_oracle:
                color = color_dict[0][(hr_count+1)*2+3]
                marker = markers[3]
                hr_count += 1
            elif "HM" in freq_oracle:
                color = color_dict[1][(hm_count+1)*2+3]
                marker = markers[2]
                hm_count += 1
            elif "FLH" in freq_oracle:
                color = color_dict[2][(flh_count+1)*2+3]
                marker = markers[1]
                flh_count += 1

            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
            ax.plot(x, means, marker=marker, label=name, color=color)
        elif "cms" in freq_oracle or "hcms" in freq_oracle:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            color = color_dict[3][0]
            name = "CM (Mean) with OUE"
            if "hcms" in freq_oracle:
                color = color_dict[3][1]
                name = "CM (Mean) with HM t=1"
            ax.plot(x, means, marker=".", label=name, color=color)
        elif "Debias Mean" in freq_oracle and "t=optimal" in freq_oracle:
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            color = color_dict[3][2]
            name = "CM (Mean) with HM t=optimal"
            ax.plot(x, means, marker=".", label=name, color=color)

    plt.xlabel("Privacy Budget Size ($\epsilon$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    ax.tick_params(axis="x", labelsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_fo_vary_eps_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_sketch_vary_d(metric="mse", same_plot=True):
    df = pd.read_csv("./metrics/group5-SR-varyD.csv")
    apple_df = pd.read_csv("./metrics/group5-apple-varyD.csv")
    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure(figsize=(12,5))
    #ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 12).as_hex()
    color_dict[1] = sns.color_palette("Purples", 12).as_hex()
    color_dict[2] = sns.color_palette("Greens", 12).as_hex()
    color_dict[3] = sns.color_palette("Reds", 3).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        hr_count = 2
        hm_count = 2
        flh_count = 0
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "FLH" in freq_oracle:
                if "HR" in freq_oracle:
                    color = color_dict[0][(hr_count+1)*2+3]
                    marker = markers[3]
                    hr_count += 1
                elif "HM" in freq_oracle:
                    color = color_dict[1][(hm_count+1)*2+3]
                    marker = markers[2]
                    hm_count += 1
                elif "FLH" in freq_oracle:
                    color = color_dict[2][(flh_count+1)*2+3]
                    marker = markers[1]
                    flh_count += 1

                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
                ax.plot(x, means, marker=marker, label=name, color=color)

        for i, freq_oracle in enumerate(apple_df["freq_oracle"].unique()):
            filtered_df = apple_df[apple_df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            ax.plot(x, means, marker="o", label=freq_oracle.upper(), color=color_dict[3][i])

        ax.set_xlabel("Domain Size ($d$)")
        ax.set_ylabel(metric_map.get(metric))
        ax.set_yscale('log')
        ax.tick_params(axis="x", labelsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout()
    plt.plot()

    if same_plot:
        plt.savefig('group5_sketch_type_vary_d.eps', format='eps', bbox_inches="tight")
    else:
        plt.savefig('group5_sketch_type_vary_d_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_sketch_vary_eps(metric="mse", same_plot=True):
    df = pd.read_csv("./metrics/group5-SR-varyEps.csv")
    apple_df = pd.read_csv("./metrics/group5-apple-varyEps.csv")
    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure(figsize=(12,5))
    #ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]
    color_dict = {}
    color_dict[0] = sns.color_palette("Blues", 12).as_hex()
    color_dict[1] = sns.color_palette("Purples", 12).as_hex()
    color_dict[2] = sns.color_palette("Greens", 12).as_hex()
    color_dict[3] = sns.color_palette("Reds", 3).as_hex()

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        hr_count = 0
        hm_count = 0
        flh_count = 0
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "HR" in freq_oracle:
                if "HR" in freq_oracle:
                    color = color_dict[0][(hr_count+1)*2+3]
                    marker = markers[3]
                    hr_count += 1
                elif "HM" in freq_oracle:
                    color = color_dict[1][(hm_count+1)*2+3]
                    marker = markers[2]
                    hm_count += 1
                elif "FLH" in freq_oracle:
                    color = color_dict[2][(flh_count+1)*2+3]
                    marker = markers[1]
                    flh_count += 1

                filtered_df = df[df["freq_oracle"] == freq_oracle]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
                ax.plot(x, means, marker=marker, label=name, color=color)

        for i, freq_oracle in enumerate(apple_df["freq_oracle"].unique()):
            filtered_df = apple_df[apple_df["freq_oracle"] == freq_oracle]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            ax.plot(x, means, marker="o", label=freq_oracle.upper(), color=color_dict[3][i])

        ax.set_xlabel("Privacy Budget Size ($\epsilon$)")
        ax.set_ylabel(metric_map.get(metric))
        ax.set_yscale('log')
        ax.tick_params(axis="x", labelsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout()
    plt.plot()

    if same_plot:
        plt.savefig('group5_sketch_type_vary_eps.eps', format='eps', bbox_inches="tight")
    else:
        plt.savefig('group5_sketch_type_vary_eps_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_sketch_type_vary_eps(metric="mse"):
    df = pd.read_csv("./metrics/group5-sketchType-varyEps.csv")
    x = np.arange(0.5, 5.5, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
        ax.plot(x, means, marker=markers[i], label=name)

    plt.xlabel("Privacy Budget Size ($\epsilon$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_sketch_type_vary_eps_' + metric + '.eps', format='eps', bbox_inches="tight")


def plot_group5_sketch_type_vary_d(metric="mse"):
    df = pd.read_csv("./metrics/group5-sketchType-varyD.csv")
    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        filtered_df = df[df["freq_oracle"] == freq_oracle]
        means = filtered_df.groupby("info", sort=False).mean()[metric]
        name = freq_oracle.replace("SR", "CM").replace("Sketch CS", "CS")
        ax.plot(x, means, marker=markers[i], label=name)

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    plt.plot()
    plt.savefig('group5_sketch_type_vary_d_' + metric + '.eps', format='eps', bbox_inches="tight")

# -----------------------------
# GROUP 6 PLOTS
# -----------------------------

def plot_group6_normalise_sketch(metric="mse", same_plot=True):
    df = pd.read_csv("./metrics/group6-normalise-sketch-FIX.csv")

    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure(figsize=(12,5))
    #ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", ".", ".", ".", ".", ".", ".", ".", "."]

    color_dict = {}
    color_dict[0] = sns.color_palette("Blues_r", 5).as_hex()
    color_dict[1] = sns.color_palette("Reds_r", 5).as_hex()
    color_dict[2] = sns.color_palette("Greens_r", 5).as_hex()
    color_dict[3] = sns.color_palette("Purples_r", 5).as_hex()
    color_dict[4] = sns.color_palette("Oranges_r", 5).as_hex()

    norms = ["None", "no norm", "prob simplex", "threshold cut", "norm"]
    figure_norms = ["no norm", "non-negative", "probability simplex", "threshold cut", "additive"]
    ignore_list = []
    only_list = ["estimator_norm=norm"]

    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            if "estimator_norm=norm" in freq_oracle or "estimator_norm=no norm" in freq_oracle:
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                if metric == "hybrid":
                    means1 = filtered_df.groupby("info", sort=False).mean()["mse"]
                    means2 = filtered_df.groupby("info", sort=False).mean()["k_mse"]
                    means = (means1 + means2) / 2
                else:
                    means = filtered_df.groupby("info", sort=False).mean()[metric]

                color = color_dict[int(i / 5)][i % 5]
                name = freq_oracle.replace("SR FLH Median", "")
                for j, norm in enumerate(norms):
                    name = name.replace("estimator_norm=" + norm, "with estimator " + figure_norms[j])
                    name = name.replace("internal_norm=" + norm, "Internal " + figure_norms[j])

                ax.plot(x, means, marker=markers[int(i / 5)], label=name, color=color)

        ax.set_xlabel("Domain Size ($d$)")
        ax.set_ylabel(metric_map.get(metric))
        ax.tick_params(axis="x", labelsize=13)
        ax.set_yscale('log')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout()
    plt.plot()

    if same_plot:
        plt.savefig('group6_normalise_sketch.eps', format='eps', bbox_inches="tight")
    else:
        plt.savefig('group6_normalise_sketch_' + metric + '.eps', format='eps', bbox_inches="tight")

def plot_group6_normalise_fo_low_d(metric="mse"):
    df = pd.read_csv("./metrics/group6-normalise-fo-lowD.csv")
    x = np.array([2 ** (i + 2) for i in range(0, 10)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    norms = ["no norm", "prob simplex", "threshold cut", "norm"]
    figure_norm = ["Non-negative Normalisation", "Probability Simplex", "Threshold Cut", "Additive Normalisation"]
    for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
        for j, norm in enumerate(norms):
            filtered_df = df[df["freq_oracle"] == freq_oracle]
            filtered_df["norm"] = df["info"].str.split("=", expand=True)[2]
            filtered_df = filtered_df[filtered_df["norm"] == norm]
            means = filtered_df.groupby("info", sort=False).mean()[metric]
            ax.plot(x, means, marker=markers[i], label=figure_norm[j])

    plt.xlabel("Domain Size ($d$)")
    plt.ylabel(metric_map.get(metric))
    ax.set_yscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.tight_layout()
    plt.plot()
    plt.savefig('group6_normalise_fo_low_d_' + metric + '.eps', format='eps', bbox_inches="tight")

def plot_group6_normalise_fo_high_d(metric="mse", same_plot=True):
    df = pd.read_csv("./metrics/group6-normalise-fo-highD-FIX2.csv")

    x = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
    fig = plt.figure(figsize=(12,4))
    #ax = fig.add_subplot(111)
    markers = ["o", ".", "^", "*", "d", "o", ">", ".", "."]

    norms = ["None", "no norm", "prob simplex", "threshold cut", "norm"]
    figure_norm = ["No Normalisation", "Non-negative Normalisation", "Probability Simplex", "Threshold Cut",
                   "Additive Normalisation"]
    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    metrics = ["mse", "k_mse"]
    for j,metric in enumerate(metrics):
        ax = axs[j]
        for i, freq_oracle in enumerate(df["freq_oracle"].unique()):
            for j, norm in enumerate(norms):
                filtered_df = df[df["freq_oracle"] == freq_oracle]
                filtered_df["norm"] = df["info"].str.split("=", expand=True)[2]
                filtered_df = filtered_df[filtered_df["norm"] == norm]
                means = filtered_df.groupby("info", sort=False).mean()[metric]
                ax.plot(x, means, marker=markers[i], label=figure_norm[j])

        ax.set_xlabel("Domain Size ($d$)")
        ax.set_ylabel(metric_map.get(metric))
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),ncol=3, fontsize=11)
    plt.tight_layout()
    plt.plot()
    if same_plot:
        plt.savefig('group6_normalise_fo_high_d.eps', format='eps', bbox_inches="tight")
    else:
        plt.savefig('group6_normalise_fo_high_d_' + metric + '.eps', format='eps', bbox_inches="tight")

def plot_hh(metric="f1"):
    df = pd.read_csv("./metrics/hh_data.csv")
    fig = plt.figure(figsize=(12,5))
    df["names"] = df["heavy_hitter"] + " " + df["freq_oracle"]

    for i, freq_oracle in enumerate(df["names"].unique()):
        filtered_df = df[df["names"] == freq_oracle]
        mean = filtered_df[metric].values.mean()
        plt.bar(freq_oracle, mean, label=freq_oracle)

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.plot()

    plt.savefig('hh.eps', format='eps', bbox_inches="tight")

def hh_table():
    df = pd.read_csv("./metrics/hh_data.csv")
    fig = plt.figure(figsize=(12,5))
    df["names"] = df["heavy_hitter"] + " " + df["freq_oracle"]
    sketch_methods = ["Mean", "Median"]

    rows_t_10 = []
    rows_t_20 = []

    for i, freq_oracle in enumerate(df["names"].unique()):
        filtered_df = df[df["names"] == freq_oracle]

        params = "T=" + freq_oracle.split("T=", )[1]
        first_part = freq_oracle.split("T=")[0]
        heavy_hitter = first_part.split(" ")[0]
        name = freq_oracle.replace(params, "").replace(heavy_hitter, "")[:-1]
        for method in sketch_methods:
            if method in params:
                params = params.replace(method, "")
                name += " (" + method + ")"

        f1 = filtered_df["f1"].values.mean()
        f1_std = filtered_df["f1"].values.std()
        recall = filtered_df["recall"].values.mean()
        precision = filtered_df["precision"].values.mean()
        if f1 != 0 and "HR" not in name and "k=16 m=128" not in params:
            if "T=10" in params:
                rows_t_10.append([heavy_hitter, name, params, precision, recall, f1, f1_std])
            else:
                rows_t_20.append([heavy_hitter, name, params, precision, recall, f1, f1_std])

    rows = [("T=10", rows_t_10), ("T=20",rows_t_20)]

    for row in rows:
        df = pd.DataFrame(row[1], columns=["Heavy Hitter", "Frequency Oracle", "Parameters", "Precision", "Recall", "F1 Score", "F1 std"])
        df = df.sort_values(by=["F1 Score"], ascending=False)
        df = df.round(3)
        print(tabulate(df, headers='keys', tablefmt='psql'))
        latex = df.to_latex(index=False)
        text_file = open(row[0] + " Table.txt", "w")
        text_file.write(latex)
        text_file.close()

def top10_hh_table_all():
    df = pd.read_csv("./metrics/final_HH_data.csv")
    fig = plt.figure(figsize=(12,5))
    df["names"] = df["heavy_hitter"] + " " + df["freq_oracle"]
    sketch_methods = ["Mean", "Median"]

    rows = []

    for i, freq_oracle in enumerate(df["names"].unique()):
        filtered_df = df[df["names"] == freq_oracle]
        params = "T=" + freq_oracle.split("T=", )[1]
        first_part = freq_oracle.split("T=")[0]
        heavy_hitter = first_part.split(" ")[0]
        name = freq_oracle.replace(params, "").replace(heavy_hitter, "")[:-1]
        for method in sketch_methods:
            if method in params:
                params = params.replace(method, "")
                name += " (" + method + ")"

        f1 = filtered_df["f1"].values.mean()
        f1_std = filtered_df["f1"].values.std()
        recall = filtered_df["recall"].values.mean()
        precision = filtered_df["precision"].values.mean()

        params = params.rstrip(" ")

        params = params.replace("k=", "r=").replace("m=", "c=")
        params = params.replace(" ", ", ")
        if "HR" not in name and "Mean" not in name and "r=16, c=128" not in params:
            rows.append([heavy_hitter, name, params, precision, recall, f1, f1_std])

    df = pd.DataFrame(rows, columns=["Heavy Hitter", "Frequency Oracle", "Parameters", "Precision", "Recall", "F1 Score", "F1 std"])
    df = df.sort_values(by=["F1 Score"], ascending=False)
    df = df.round(3)
    df["F1 Score"] = df["F1 Score"].astype("str") + " (" + df["F1 std"].astype("str") + ")"
    df = df.drop("F1 std", axis=1)
    df.index = np.arange(1, len(df) + 1)
    print("Length of DF:", len(df))
    print(tabulate(df, headers='keys', tablefmt='psql'))
    latex = df.to_latex(index=True)
    text_file = open("Top 10 HH (all sketches) Table.txt", "w")
    text_file.write(latex)
    text_file.close()

def top10_hh_table():
    df = pd.read_csv("./metrics/final_HH_data.csv")
    fig = plt.figure(figsize=(12,5))
    df["names"] = df["heavy_hitter"] + " " + df["freq_oracle"]
    sketch_methods = ["Mean", "Median"]

    rows = []

    for i, freq_oracle in enumerate(df["names"].unique()):
        filtered_df = df[df["names"] == freq_oracle]
        params = "T=" + freq_oracle.split("T=", )[1]
        first_part = freq_oracle.split("T=")[0]
        heavy_hitter = first_part.split(" ")[0]
        name = freq_oracle.replace(params, "").replace(heavy_hitter, "")[:-1]
        for method in sketch_methods:
            if method in params:
                params = params.replace(method, "")
                name += " (" + method + ")"


        if "FLH" in name:
            sketch_method = "CM (Median)"
            oracle = "FLH"
        elif "HCMS" in name:
            sketch_method = "CM (Mean)"
            oracle = "HM (t=1)"
        else:
            sketch_method = "CM (Mean)"
            oracle = "OUE"

        f1 = filtered_df["f1"].values.mean()
        f1_std = filtered_df["f1"].values.std()
        recall = filtered_df["recall"].values.mean()
        precision = filtered_df["precision"].values.mean()

        params = params.rstrip(" ")

        params = params.replace("k=", "r=").replace("m=", "c=")
        params = params.replace(" ", ", ")
        if f1 != 0 and "HR" not in name and "r=16, c=128" not in params and "Mean" not in name:
            rows.append([heavy_hitter, sketch_method, oracle, params, precision, recall, f1, f1_std])

    df = pd.DataFrame(rows, columns=["Heavy Hitter", "Sketch Method", "Frequency Oracle", "Parameters", "Precision", "Recall", "F1 Score", "F1 std"])
    df = df.sort_values(by=["F1 Score"], ascending=False)
    df = df.round(3)
    df["F1 Score"] = df["F1 Score"].astype("str") + " (" + df["F1 std"].astype("str") + ")"
    df = df.drop("F1 std", axis=1)
    df = df.head(20)
    df.index = np.arange(1, len(df) + 1)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    latex = df.to_latex(index=True)
    text_file = open("Top 10 HH Table.txt", "w")
    text_file.write(latex)
    text_file.close()

def clean_hh():
    df = pd.read_csv("./metrics/hh_data2.csv")
    fig = plt.figure(figsize=(12,5))
    df["names"] = df["heavy_hitter"] + " " + df["freq_oracle"]

    filtered_df = df[(~df["names"].str.contains("TreeHist")) & (~df["names"].str.contains("HR"))
                     & (~df["names"].str.contains("SFP CM FLH"))
                     & (~df["names"].str.contains("k=16 m=128")) & (~df["names"].str.contains("Mean"))]
    print(filtered_df["names"].unique())
    print(len(filtered_df["names"].unique()))
    filtered_df = filtered_df.drop("names", axis=1)
    filtered_df.to_csv("./final_HH_data.csv", index=False)


# -----------------------------
# Running plots...
# -----------------------------

#plot_hh()
#hh_table()

# Table 2
top10_hh_table_all()
top10_hh_table()

#clean_hh()

# plot_group5_sketch_vary_eps("mse")

# -----------------------------
# GROUP 1 - Direct Encoding
# -----------------------------

plot_group1_vary_eps() # Fig 1a
plot_group1_vary_d() # Fig 1b
plot_group1_time() # Fig 1c

# -----------------------------
# GROUP 2 - Local Hashing
# -----------------------------

plot_group2_vary_eps() # Fig 3a
plot_group2_vary_d() # Fig 3b
plot_group2_time() # Fig 3c
plot_group2_k() # Figure 2

# -----------------------------
# GROUP 3 - HADAMARD
# -----------------------------

plot_group3_vary_T() # Fig 4a
plot_group3_vary_eps() # Fig 4b
plot_group3_vary_d() # Fig 4c
plot_group3_lowEps()

# -----------------------------
# GROUP 4 - Initial Comparisons
# -----------------------------

plot_group4_vary_eps() # Fig 5a
plot_group4_vary_d() # Fig 5b
plot_group4_time() # Fig 5c

# ----------------------------
# GROUP 5 - Sketching
# -----------------------------
# plot_group5_vary_m()
# plot_group5_vary_k()
# plot_group5_SR_vary_m()

plot_group5_SR_vary_m_individual("mse") # Fig 7a
plot_group5_SR_vary_m_individual("k_mse") # Fig 7b

# plot_group5_SR_vary_k()
plot_group5_SR_vary_k_individual("k_mse") # Fig 7c

plot_group5_bloom_comparison() # Fig 8

# plot_group5_all_vary_eps(metric="mse")
# plot_group5_all_vary_eps(metric="k_mse")
# plot_group5_all_vary_d(metric="mse")
# plot_group5_all_vary_d(metric="k_mse")

plot_group5_fo_vary_d() # Figure 9
plot_group5_fo_vary_eps() # Figure 10

# plot_group5_debias()

#plot_group5_sketch_vary_d()
#plot_group5_sketch_vary_eps()

# plot_group5_sketch_type_vary_eps("mse")
# plot_group5_sketch_type_vary_eps("k_mse")
# plot_group5_sketch_type_vary_d("mse")
# plot_group5_sketch_type_vary_d("k_mse")

plot_group5_rappor("mse") # Fig 6

# -----------------------------
# GROUP 6 - Normalisation Experiments
# -----------------------------

#plot_group6_normalise_sketch("mse")
#plot_group6_normalise_sketch("k_mse")

plot_group6_normalise_fo_high_d("mse") # Figure 11

# plot_group6_normalise_fo_low_d("mse")
# plot_group6_normalise_fo_low_d("k_mse")