# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:16:14 2022

@author: olijm
"""
import WordleCracker as WC
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

calcTimes,scoreSet = WC.calcGuessTimeDist(200)

plt.figure()
plt.hist(calcTimes,bins=np.arange(8)+0.5,orientation='horizontal',color = [0.2,0.2,0.2],rwidth=0.9)
plt.ylabel('Number of guesses to reach solution')
plt.xlabel('Frequency')

ax = plt.gca()
ax.invert_yaxis()

print(np.mean(calcTimes))

renderImg = WC.renderScoreSet(scoreSet,10)
Img255 = np.ceil(renderImg*255)
outImg = im.fromarray(Img255.astype('uint8'),'RGB')

outImg.save('C:\\Users\\olijm\\Desktop\\Wordle\\ImageTest.png')