# -*- coding: utf-8 -*-
"""
.. module: photosynthesis
    :synopsis: Wrapper class for running different photosynthesis models.
.. moduleauthor:: Olli-Pekka Tikkasalo
"""

import numpy as np
import logging
from typing import List, Dict, Tuple


# from pyAPES.microclimate.micromet import e_sat, latent_heat
# from pyAPES.leaf.boundarylayer import leaf_boundary_layer_conductance
from pyAPES.leaf.photo import photo_c3_medlyn_farquhar, photo_c3_medlyn_farquhar_gm
# from pyAPES.utils.constants import DEG_TO_KELVIN, PAR_TO_UMOL, EPS, SPECIFIC_HEAT_AIR, \
#     GAS_CONSTANT, O2_IN_AIR, MOLAR_MASS_H2O

# H2O_CO2_RATIO = 1.6  # H2O to CO2 diffusivity ratio [-]
# TN = 25.0 + DEG_TO_KELVIN  # reference temperature, 283.15 [K]

logger = logging.getLogger(__name__)


class Photosyntehsis_model():

    def __init__(self, photo_model: str):
        self.photo_model = photo_model

        if self.photo_model.upper() == 'MEDLYN_FARQUHAR':
            self.output_names = ['An', 'Rd', 'fe', 'gs_opt', 'Ci', 'Cs']
        elif self.photo_model.upper() == 'MEDLYN_FARQUHAR_GM':
            self.output_names = ['An', 'Rd', 'fe',
                                'gs_opt', 'Ci', 'Cs', 'gm', 'Cc']

    def run(self, forcing, photop):
        if self.photo_model.upper() == 'MEDLYN_FARQUHAR':
            results = photo_c3_medlyn_farquhar(photop,
                                               forcing['Qp'], forcing['T'], forcing['VPD'],
                                               forcing['co2'], forcing['gb_c'], forcing['gb_v'], P=forcing['air_pressure'])
            results_dict = {k: v for k, v in zip(self.output_names, results)}
            results_dict['gm'] = None
            results_dict['Cc'] = None

        elif self.photo_model.upper() == 'MEDLYN_FARQUHAR_GM':
            results = photo_c3_medlyn_farquhar_gm(photop, forcing['Qp'], forcing['Qa'], forcing['T'],
                                                  forcing['VPD'], forcing['co2'], forcing['gb_c'], forcing['gb_c'],
                                                  P=forcing['air_pressure'])
            results_dict = {k: v for k, v in zip(self.output_names, results)}
        return results_dict


def initialize_photo_forcing(num_elements):
    forcing = {}
    forcing['Qp'] = np.zeros((num_elements,))*np.nan
    forcing['Qa'] = np.zeros((num_elements,))*np.nan
    forcing['T'] = np.zeros((num_elements,))*np.nan
    forcing['VPD'] = np.zeros((num_elements,))*np.nan
    forcing['co2'] = np.zeros((num_elements,))*np.nan
    forcing['gb_c'] = np.zeros((num_elements,))*np.nan
    forcing['gb_v'] = np.zeros((num_elements,))*np.nan

    return forcing


def set_photo_forcing(forcing, Qp, T, VPD, co2, gb_c, gb_v, P, Qa):
    if Qa is not None:
        forcing['Qa'] = Qa
    keys = ['Qp', 'T', 'VPD', 'co2', 'gb_c', 'gb_v', 'air_pressure']
    values = (Qp, T, VPD, co2, gb_c, gb_v, P)

    for key, val in zip(keys, values):
        forcing[key] = val
    return forcing
