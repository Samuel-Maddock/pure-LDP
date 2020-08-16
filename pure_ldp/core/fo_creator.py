from pure_ldp.frequency_oracles import *
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


def _create_pure_fo_instance(obj_type, name, obj_params):
    """
    Used internally to create instances of various Client/Server frequency oracles

    Args:
        obj_type (str): Either "Client" or "Server"
        name: Name of the frequency oracle to create
        obj_params: Parameters for the frequency oracle object

    Returns: Instance of name + obj_type frequency oracle

    """
    fo_list = class_list[obj_type]

    split = name.split("_")  # Get prefix of client name i.e if passed "local_hashing" get "LH" as prefix

    if len(split) > 1:
        name = ""
        for word in split:
            name += word[0]

    name = name.upper()
    if name == "HR": name = "HADAMARDRESPONSE"

    upper_fo_list = list(map(lambda x: x.upper(), fo_list))

    if name not in upper_fo_list:
        raise ValueError("Frequency oracle must be one of:", fo_list,
                         "\n NOTE: Values are case insensitive")

    fo_name = client_class_list[upper_fo_list.index(name)] + obj_type

    constructor = globals().get(fo_name)
    expected_params = list(inspect.signature(constructor).parameters)

    params = dict(
        (key.split("=")[0], obj_params[key.split("=")[0]]) for key in expected_params if key in obj_params.keys())

    return constructor(**params)


def create_fo_client_instance(name, client_params):
    """
    Given a name of a frequency oracle creates a client instance of it

    Args:
        name: Name of frequency oracle (i.e LH, HE)
        client_params: The parameters for the client frequency oracle object

    Returns: A frequency oracle instance of nameClient

    """
    return _create_pure_fo_instance("Client", name, client_params)


def create_fo_server_instance(name, server_params):
    """
    Given a name of a frequency oracle creates a server instance of it

    Args:
        name: Name of frequency oracle (i.e LH, HE)
        server_params: The parameters for the server frequency oracle

    Returns: A frequency oracle instance of nameServer

    """
    return _create_pure_fo_instance("Server", name, server_params)
