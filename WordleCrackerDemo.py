# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:40:21 2022

@author: olijm
"""
import WordleCracker as WC

tgt = 'shire'
guess1 = 'tares'

#Read in starting data
letterStore = WC.getData();

score1 = WC.scoreGuess(WC.word2line(tgt), WC.word2line(guess1))

#Narrow down the word list based on first guess
letterStore = WC.runGuessCycleManual(guess1,score1,letterStore)

guess2 = 'spore'
score2 = WC.scoreGuess(WC.word2line(tgt), WC.word2line(guess2))

letterStore = WC.runGuessCycleManual(guess2,score2,letterStore)

guess3 = 'apnea'
score3 = WC.scoreGuess(WC.word2line(tgt), WC.word2line(guess3))

letterStore = WC.runGuessCycleManual(guess3,score3,letterStore)