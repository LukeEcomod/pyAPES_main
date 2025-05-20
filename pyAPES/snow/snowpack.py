# -*- coding: utf-8 -*-
"""
.. module: snowpack
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Snowpack model driver calls for degreeday or fsm2 model*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.snow.pyFSM2.fsm2_standalone import FSM2
from pyAPES.snow.degreeday.degreeday import DegreeDaySnow
import logging

logger = logging.getLogger(__name__)

EPS = np.finfo(float).eps  # machine epsilon

class Snowpack(object):
    def __init__(self, snow_model, snowpara) -> object:
        """
        Args:
        Returns:
        """
        
        if snow_model['type'] == 'degreeday':
            self.snowpack = DegreeDaySnow(snowpara)

        elif snow_model['type'] == 'fsm2':
            self.snowpack = FSM2(snowpara)

        else:
            raise NotImplementedError(f"Snow model type is not implemented.")
            # importataan loggeri ja logger error (katso muista moduuleista mallia)
            # esim heat.py stä

    def run(self, dt: float, forcing: Dict) -> Tuple:
        """
        Args:
        Returns:
        """

        fluxes, states = self.snowpack.run(dt, forcing)
        
        return fluxes, states

# EOF