# Pure-LDP

pure-LDP is a Python package that provides simple implementations of various state-of-the-art LDP algorithms (both Frequency Oracles and Heavy Hitters) with the main goal of providing a single, simple interface to benchmark and experiment with these algorithms.

pure-LDP started as a package for pure LDP frequency oracles detailed in the paper 
["Locally Differentially Private Protocols for Frequency Estimation"](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao) by Wang et al but has evolved to implement many other state-of-the art LDP frequency estimation protocols like Apple's CMS/HCMS and Google's RAPPOR. It also contains implementations of various heavy hitter protocols like Apple's Sequence Fragment Puzzle (SFP) and Prefix Extending Method (PEM).

The main goal of the library is to develop a framework that allows easy use of frequency oracles (FOs) for benchmarking and experimentation, easy extension to implement new oracles and the flexibility to swap out FOs in current protocols (i.e mixing and matching FOs with heavy hitter protocols).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install pure-ldp
```
To upgrade to the [latest version](https://pypi.org/project/pure-ldp/)
```bash
pip install pure-ldp --upgrade
```
Requires numpy, scipy, xxhash, bitarray and bitstring. For simulation plots, matplotlib and seaborn are required.


 ## Outline

The package has implementations of all three main frequency oracles detailed in paper 
["Locally Differentially Private Protocols for Frequency Estimation"](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao) by Wang et al which are:
1. (Optimal) Unary Encoding - Under ```pure_ldp.frequency_oracles.unary_encoding``` 
2. (Summation/Thresholding) Histogram encoding - Under ```pure_ldp.frequency_oracles.histogram_encoding``` 
3. (Optimal) Local Hashing - Under ```pure_ldp.frequency_oracles.local_hashing```

The package also includes an implementation of the heavy hitter algorithm Prefix Extending Method (PEM) under ```pure_ldp.heavy_hitters.prefix_extending```

Over time it has evolved to include many more implementations of other LDP frequency estimation algorithms:
1. Apple's [Count Mean Sketch (CMS / HCMS)](https://machinelearning.apple.com/research/learning-with-privacy-at-scale) Algorithm - This is under ```pure_ldp.frequency_oracles.apple_cms```
2. Google's RAPPOR i.e DE combined with Bloom filters under ```pure_ldp.frequency_oracles.rappor```
3. [Hadamard Response (HR)](https://arxiv.org/abs/1802.04705) - This is under ```pure_ldp.frequency_oracles.hadamard_response``` the code implemented for this is simply a pure-LDP wrapper of the repo [hadamard_response](https://github.com/zitengsun/hadamard_response)
4. Hadamard Mechanism (HM) under ```pure_ldp.frequency_oracles.hadamard_mechanism```
5. Direct Encoding (DE) / Generalised Randomised Response under ```pure_ldp.frequency_oracles.direct_encoding```
6. Fast Local Hashing (FLH) a heuristic variant of OLH under ```pure_ldp.frequency_oracles.local_hashing```
7. Generic private sketching protocols (SketchResponse) under ```pure_ldp.frequency_oracles.sketch_response```

The library also includes implementations of other Heavy Hitter (HH) algorithms:
 1. Apple's Sequence Fragment Puzzle (SFP) algorithm under ```pure_ldp.frequency_oracles.apple_sfp```
 2. TreeHistogram (by [Bassily et al](https://arxiv.org/abs/1707.04982)) under ```pure_ldp.frequency_oracles.treehistogram```
 
## Basic Usage

```python
import numpy as np
from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer

# Using Optimal Local Hashing (OLH)

epsilon = 3 # Privacy budget of 3
d = 4 # For simplicity, we use a dataset with 4 possible data items

client_olh = LHClient(epsilon=epsilon, d=d, use_olh=True)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)

# Test dataset, every user has a number between 1-4, 10,000 users total
data = np.concatenate(([1]*4000, [2]*3000, [3]*2000, [4]*1000))

for item in data:
    # Simulate client-side privatisation
    priv_data = client_olh.privatise(item)

    # Simulate server-side aggregation
    server_olh.aggregate(priv_data)

# Simulate server-side estimation
print(server_olh.estimate(1)) # Should be approximately 4000 +- 200

```

See [example.py](https://github.com/Samuel-Maddock/pure-LDP/blob/master/example.py) for more examples.


 ## Simulation Framework
 
 This is currently WIP but there is already significant code under ```pure_ldp.simulations``` that allow you to build experiments to compare various frequency oracles/heavy hitters under various conditions. Generic helpers to run experiments for FOs and HHs are included under ```pure_ldp.simulations.helpers```. See ```pure_ldp.simulations.paper_experiments.py``` for examples
 

## TODO
1. Better documentation !

## Acknowledgements

1. Some OLH code is based on the implementation by [Tianhao Wang](https://github.com/vvv214): [repo](https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)
2. The Hadamard Response code is just a wrapper of the k2khadamard.py code in the repo [hadamard_response](https://github.com/zitengsun/hadamard_response) by [Ziteng Sun](https://github.com/zitengsun)

## Contributing
If you feel like this package could be improved in any way, open an issue or make a pull request!


## License
[MIT](https://choosealicense.com/licenses/mit/)
