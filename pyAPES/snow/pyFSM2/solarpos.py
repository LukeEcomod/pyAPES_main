"""
.. module:: solarpos
    :synopsis: Solar position calculation component
.. moduleauthor:: Jari-Pekka Nousu

*Calculates the azimuth and elevation angles of the sun (based on FSM2)*
"""

import numpy as np
from math import sin, asin, cos, acos
from pyAPES.utils.constants import PI

class SolarPos():
    def __init__(self):
        """
        Initialize the SolarPos model.
        """

    def run(self, dt: float, forcing: dict):
        """
        Calculate the solar azimuth and elevation angles for a given timestep.

        Args:
            dt (float): timestep (s)
            forcing (dict):
                'year' (int): Current year.
                'month' (int): Current month (1-12).
                'day' (int): Current day of the month.
                'hour' (float): Current time in decimal hours (e.g., 13.5 for 1:30 PM).
                'lat' (float): Latitude in radians.
                'noon' (float): Local solar noon in decimal hours.

        Returns:
            tuple:
                fluxes (dict):
                states (dict):
                    'azimuth' (float): Solar azimuth angle in radians.
                    'elevation' (float): Solar elevation angle in radians.
        """

        year = forcing['year']
        month = forcing['month']
        day = forcing['day']
        hour = forcing['hour']
        lat = forcing['lat']
        noon = forcing['noon']

        DoY = (7*year)/4 - 7*(year+(month+9)/12)/4 + (275*month)/9 + day - 30

        dangle = 2*PI*(DoY - 1)/365

        declin = (0.006918 - 0.399912*cos(dangle)   + 0.070257*sin(dangle)
                        - 0.006758*cos(2*dangle) + 0.000907*sin(2*dangle)
                        - 0.002697*cos(3*dangle) + 0.001480*sin(3*dangle))
        eqtime = ((0.000075 + 0.001868*cos(dangle)   - 0.032077*sin(dangle)
                        - 0.014615*cos(2*dangle) - 0.04089*sin(2*dangle))
                *(12/PI))
        
        hangle = (PI/12)*(noon - hour - eqtime)

        elev = asin(sin(declin)*sin(lat) + cos(declin)*cos(lat)*cos(hangle))

        azim = acos((sin(elev)*sin(lat) - sin(declin))/(cos(elev)*cos(lat)))

        if (hangle < 0):
               azim = - azim

        fluxes = {}
        states = {'azimuth': azim, 'elevation': elev}
    
        return fluxes, states