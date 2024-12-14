import examples.gaussian_model.environments.gaussian_squeeze as gaussian_squeeze

"""
 Creates a new environment.
"""
def make(params):
    domain_name = params["domain_name"]
    if domain_name.startswith("GaussianSqueeze-"):
        params["nr_epochs"] = 10000
        return gaussian_squeeze.make(params)
    raise ValueError("Unknown domain '{}'".format(domain_name))