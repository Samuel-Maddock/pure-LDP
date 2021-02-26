import sys

from pure_ldp.heavy_hitters.prefix_extending import *
from pure_ldp.core.hh_creator import *
from pure_ldp.core.fo_creator import *

import time

class PureHHSimulation():
    def __init__(self, hh_name, params):
        super().__init__()

        self.k = params.get("k", None)
        self.threshold = params.get("threshold", None)
        self.name = hh_name

        self.freq_oracle = params.get("freq_oracle")
        self.freq_oracle_params = params.get("freq_oracle_params")

        self.freq_oracle_server_params = self.freq_oracle_params["server_params"]
        self.freq_oracle_client_params = self.freq_oracle_params["client_params"]

        self.hh_client_params = params["hh_client_params"]
        self.hh_server_params = params["hh_server_params"]

        self.hh_client_params["alphabet"] = params["alphabet"]
        self.hh_client_params["alphabet"] = params["alphabet"]

        self.memory_safe = params.get("memory_safe", False)

        if len(set(params["alphabet"]) - {"0", "1"}) != 0: # TODO: Rework this
            index_mapper = lambda x: x
            self.hh_client_params["index_mapper"] = index_mapper
            self.hh_server_params["index_mapper"] = index_mapper
        else:
            # NOTE: Setting the index_mapper to None will default index mappers to calculating numbers from binary strings in HeavyHitter objects
            index_mapper = None
            self.hh_client_params["index_mapper"] = index_mapper
            self.hh_server_params["index_mapper"] = index_mapper

        self.client = None
        self.server = None

        if self.freq_oracle is not None and self.freq_oracle_server_params is not None and self.freq_oracle_client_params is not None:
            self.server = create_fo_server_instance(self.freq_oracle, self.freq_oracle_server_params)

            try: # Some freq_oracles need to pass hash funcs from server to client
                hash_funcs = self.server.get_hash_funcs()
                self.freq_oracle_client_params["hash_funcs"] = hash_funcs
            except AttributeError:
                pass

            try: # Sketching oracles require the hash functions to be reset
                server_fo_hash_funcs = self.server.server_fo_hash_funcs
                self.freq_oracle_client_params["server_fo_hash_funcs"] = server_fo_hash_funcs
            except AttributeError:
                pass

            self.client = create_fo_client_instance(self.freq_oracle, self.freq_oracle_client_params)

    def run(self, data):

        # Client-side
        start_time = time.time()

        # If using a frequency oracle that outputs with arrays then for large data privatising then storing can cause memory issues
            # If self.memory_safe then data is privatised and aggregated at the same time and not stored in memory

        if not self.memory_safe:
            self.hh_client_params["fo_client"] = self.client
            client = create_hh_client_instance(self.name, self.hh_client_params)

            ldp_data = []
            print(len(data))
            for item in data:
                priv = client.privatise(item)
                ldp_data.append(priv)

            print("Client HH Privatising Finished...")
            client_time = time.time() - start_time
            start_time = time.time()

            # Server-side
            self.hh_server_params["fo_server"] = self.server

            server = create_hh_server_instance(self.name, self.hh_server_params)

            # Aggregation
            for data in ldp_data:
                server.aggregate(data)
        else:
            self.hh_client_params["fo_client"] = self.client
            client = create_hh_client_instance(self.name, self.hh_client_params)

            client_time = 0

            # Server-side
            self.hh_server_params["fo_server"] = self.server
            server = create_hh_server_instance(self.name, self.hh_server_params)

            # Aggregation
            for item in data:
                server.aggregate(client.privatise(item))

        # PEM uses top T and SFP/TreeHist uses threshold
        output = []

        heavy_hitters, frequencies = server.find_heavy_hitters(k=self.k, threshold=self.threshold)

        output = list(zip(heavy_hitters, frequencies))

        # for item in D:
        #     freq = freq_oracle.estimate(item)
        #     if freq >= self.threshold*math.sqrt(freq_oracle.n):
        #         output.append((item, freq))

        output.sort(reverse=False, key=lambda x: x[1])

        server_time = time.time() - start_time

        print(output)
        print(len(output))

        return output, client_time, server_time
