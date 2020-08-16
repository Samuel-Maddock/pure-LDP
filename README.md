# Pure-LDP

pure-LDP is a Python package that provides simple implementations of various state-of-the-art LDP algorithms (both Frequency Oracles and Heavy Hitters) with the main goal of providing a single, simple interface to use these algorithms.

pure-LDP started as a package for pure LDP frequency oracles detailed in the paper 
["Locally Differentially Private Protocols for Frequency Estimation"](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao) by Wang et al


The package has implementations of all three main frequency oracles detailed in that paper:
1. (Optimal) Unary Encoding - Under ```pure_ldp.frequency_oracles.unary_encoding``` 
2. (Summation/Thresholding) Histogram encoding - Under ```pure_ldp.frequency_oracles.histogram_encoding``` 
3. (Optimal) Local Hashing - Under ```pure_ldp.frequency_oracles.local_hashing```

The package also includes an implementation of the heavy hitter algorithm Prefix Extending Method (PEM)
* This is under ```pure_ldp.heavy_hitters.prefix_extending```

The package also contains other LDP implementations:
1. Apple's [Count Mean Sketch (CMS / HCMS)](https://machinelearning.apple.com/research/learning-with-privacy-at-scale) Algorithm - This is under ```pure_ldp.frequency_oracles.apple_cms```
2. [Hadamard Response (HR)](https://arxiv.org/abs/1802.04705) - This is under ```pure_ldp.frequency_oracles.hadamard_response```the code implemented for this is simply a pure-LDP wrapper of the repo [hadamard_response](https://github.com/zitengsun/hadamard_response)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install pure-ldp
```
To upgrade to the [latest version](https://pypi.org/project/pure-ldp/)
```bash
pip install pure-ldp --upgrade
```
Requires numpy, scipy, xxhash, bitarray and bitstring

## Usage

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

## TODO
1. Implementation of Apple's SFP
2. Implementation of Google's RAPPOR
3. Implementation of frequency oracles/heavy hitter algorithms detailed in 
4. Better documentation !

## Acknowledgements

1. Some OLH code is based on the implementation by [Tianhao Wang](https://github.com/vvv214): [repo](https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)
2. The Hadamard Response code is just a wrapper of the k2khadamard.py code in the repo [hadamard_response](https://github.com/zitengsun/hadamard_response) by [Ziteng Sun](https://github.com/zitengsun)

## Contributing
If you feel like this package could be improved in any way, open an issue or make a pull request!


## License
[MIT](https://choosealicense.com/licenses/mit/)
