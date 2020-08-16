from pure_ldp.heavy_hitters import *
import copy
import inspect


# Used to create a list of possible frequency oracles in the pure-LDP library

client_class_list = []
server_class_list = []
globs = list(globals().keys()).copy()  # Create copy, since globals updates too much to iterate through

for key in globs:
    if "Client" in key:
        client_class_list.append(key.replace("Client", ""))
    if "Server" in key:
        server_class_list.append(key.replace("Server", ""))

class_list = {"Client": client_class_list, "Server": server_class_list}


def _create_hh_instance(obj_type, name, obj_params):
    """
    Used internally to create a heavy hitter instance (client/server)

    Args:
        obj_type: Either "Client" or "Server"
        name: Name of heavy hitter algorithm
        obj_params: Parameters for the client/server object

    Returns: name + obj_type heavy hitter object

    """
    fo_list = class_list[obj_type]

    split = name.split("_")  # Get prefix of client name i.e if passed "local_hashing" get "LH" as prefix

    if len(split) > 1:
        name = ""
        for word in split:
            name += word[0]

    name = name.upper()
    if name == "HR": name = "HADAMARDRESPONSE"

    upper_hh_list = list(map(lambda x: x.upper(), fo_list))

    if name not in upper_hh_list:
        raise ValueError("Heavy Hitter", obj_type, "must be one of:", fo_list,
                         "\n NOTE: Values are case insensitive")

    hh_name = client_class_list[upper_hh_list.index(name)] + obj_type

    constructor = globals().get(hh_name)
    expected_params = list(inspect.signature(constructor).parameters)

    params = dict(
        (key.split("=")[0], obj_params[key.split("=")[0]]) for key in expected_params if key in obj_params.keys())

    return constructor(**params)


def create_hh_client_instance(name, client_params):
    """
    Returns a heavy hitter client given the name and parameters

    Args:
        name: Name of heavy hitter algorithm (i.e PEM)
        client_params: The parameters for the client

    Returns: nameClient heavy hitter object

    """
    return _create_hh_instance("Client", name, client_params)


def create_hh_server_instance(name, server_params):
    """
    Returns a heavy hitter server given the name and parameters

    Args:
        name: Name of heavy hitter algorithm (i.e PEM)
        server_params: The parameters for the server

    Returns: nameServer heavy hitter object

    """
    return _create_hh_instance("Server", name, server_params)
