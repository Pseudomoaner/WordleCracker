# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:16:14 2022

@author: olijm
"""
import WordleCracker as WC
import numpy as np
import matplotlib.pyplot as plt

calcTimes = WC.calcGuessTimeDist(1000)

plt.figure()
plt.hist(calcTimes,bins=np.arange(15)+0.5)

print(np.mean(calcTimes))