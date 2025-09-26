# -*- coding: utf-8 -*-
"""
.. module: snowpack
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Snowpack model driver calls for degreeday or fsm2 model*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.snow.pyFSM2.fsm2_coupled import FSM2
from pyAPES.snow.degreeday.degreeday import DegreeDaySnow
import logging

logger = logging.getLogger(__name__)

EPS = np.finfo(float).eps  # machine epsilon

class Snowpack(object):
    def __init__(self, snowpara) -> object:
        """
        Args:
        Returns:
        """
        
        if snowpara['snow_model'] == 'degreeday':
            self.snowpack = DegreeDaySnow(snowpara['degreeday'])

        elif snowpara['snow_model'] == 'fsm2':
            self.snowpack = FSM2(snowpara['fsm2'])

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