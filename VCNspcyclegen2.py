# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:34:47 2015

@author: Alex
"""

def VCNspcyclegen2(freq):
    import numpy as np;
    
    freq = np.array(freq);
    spsec = np.ones((1,len(freq)))*128;
    spsec = np.array(spsec);
    spsec = spsec[0];
    spcycle = spsec/freq;
    return spcycle, spsec
    
