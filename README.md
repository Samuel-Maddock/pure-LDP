# Pure-LDP

pure-LDP is a Python package that provides simple implementations of pure LDP frequency oracles detailed in the paper 
["Locally Differentially Private Protocols for Frequency Estimation"](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao) by Wang et al

The package has implementations of all three main techniques:
1. Unary Encoding - Under ```pure_ldp.local_hashing.ue_client``` and ```pure_ldp.local_hashing.ue_server``` 
2. Histogram encoding - Under ```pure_ldp.local_hashing.lh_client``` and ```pure_ldp.local_hashing.lh_sever``` 
3. Local Hashing - Under ```pure_ldp.local_hashing.he_client``` and ```pure_ldp.local_hashing.he_server``` 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install pure-ldp
```

Requires numpy, scipy and xxhash.

## Usage

```python
import numpy as np
from pure-ldp.local_hashing.lh_client import LHClient
from pure-ldp.local_hashing.lh_server import LHServer

# Using Optimal Local Hashing (OLH)

epsilon = 3 # Privacy budget of 3
d = 4 # For simplicity, we use a simple dataset with 4 possible data items

client_olh = LHClient(epsilon=epsilon, use_olh=True)
server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)

# Test dataset, every user has a number between 1-4, 10,000 users total
data = np.concatenate(([1]*4000, [2]*3000, [3]*2000, [4]*1000))

for index, item in enumerate(data):
    # Simulate client-side process
    print(item)
    priv_data = client_olh.privatise(item, index) # We use the user's index as a hash seed, in practice this should be randomly+uniquely generated
    print(priv_data)

    # Simulate server-side aggregation
    server_olh.aggregate(priv_data, index)

# Simulate server-side estimation
print(server_olh.estimate(1)) # Should be approximately 4000 +- 100

```

Checkout [example.py](https://github.com/Samuel-Maddock/pure-LDP/blob/master/example.py) for more examples.

## TODO
1. More documentation
2. Implementation of PEM

## Contributing
If you feel like this package could be improved in any way, open an issue or make a pull request!


## License
[MIT](https://choosealicense.com/licenses/mit/)