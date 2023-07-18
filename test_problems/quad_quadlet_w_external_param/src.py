import torch

from ..accelerator_toy_models import simple_lattices


def beamsize_quadlet_w_external_param(input_data):
    """
    calculate final beam sizes as a function of quadrupole strengths [k1,k2,k3]

    K : magnetic strength of each quad magnet, torch.tensor, shape (3,)
    noise : rms noise level of measurement, float (default 0.0)

    """
    # process inputs
    K = torch.tensor([input_data[f"k{i}"] for i in range(1, 5)])
    s = torch.tensor(input_data["emittance_scale"])

    # generate initial beam matrix
    # - z geometric emittance of 1.0e-8 m-rad
    init_beam_matrix = torch.eye(6) * 1.0e-8

    # set x_rms beam size to 1 mm and rms divergence to 0.1 mrad
    # note minimum geometric emittance is 1e-7 m-rad
    init_beam_matrix[0, 0] = (1.0e-3 * s) ** 2
    init_beam_matrix[1, 1] = 1.0e-4 ** 2
    init_beam_matrix[2, 2] = 1.0e-3 ** 2
    init_beam_matrix[3, 3] = 1.0e-4 ** 2

    #print(torch.det(init_beam_matrix[:2, :2]).sqrt())

    # create accelerator lattice object with one quad and a drift
    line = simple_lattices.create_quadlet(K)

    # propagate beam matrix
    final_beam_matrix = line.propagate_beam_matrix(init_beam_matrix)

    #print(torch.det(final_beam_matrix[:2, :2]).sqrt())

    total_size = torch.sqrt(final_beam_matrix[0, 0] + final_beam_matrix[2, 2])
    return {
        "m11": float(final_beam_matrix[0, 0]),
        "m22": float(final_beam_matrix[2, 2]),
        "total_size": float(total_size)
    }


# define vocs
VARIABLES = {"k1": [-300, 300], "k2": [-300, 300], "k3": [-300, 300], "k4": [-300, 300]}
OBJECTIVES = {"total_size": "MINIMIZE"}

