# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:39:19 2022

@author: olijm
"""
import csv
import numpy as np
import re

###### Core functions ######

# Function that reads in the raw data and converts to a numeric numpy array
def getData():
    inFile = './inWords.csv'
    
    with open(inFile, mode ='r')as file:
   
        # reading the CSV file
        csvFile = csv.reader(file)
         
        dataStore = np.array([0,0,0,0,0])
        # Fill up data storage
        for lines in csvFile:
            dataStore = np.vstack([dataStore,word2line(lines[0])])
    
    return dataStore[1:,:]

def getTgts():
    inFile = './tgtWords.csv'
    
    with open(inFile, mode ='r')as file:
   
        # reading the CSV file
        csvFile = csv.reader(file)
         
        dataStore = np.array([0,0,0,0,0])
        # Fill up data storage
        for lines in csvFile:
            dataStore = np.vstack([dataStore,word2line(lines[0])])
    
    return dataStore[1:,:]


#Function that converts a line of the numpy array back into a human-readable word
def line2word(npLine):
    return ''.join(chr(c) for c in npLine)

#Function that converts a string into a numpy-compatible list
def word2line(string):
    return np.array([ord(c) for c in string])

#Function that detects if a new guess is discounted by a previous one
def permitGuess(guessOld,outcome,guessNew):
    green = outcome == 2
    yellow = outcome == 1
    grey = outcome == 0
    
    #Are there any repeated letters in the old guess where some come out yellow and some come out grey? If so, sets an upper limit on the number of repeats of said letter.
    maxReps = np.zeros(np.shape(yellow)) #The maximum number of times this letter can appear in the new word (incl. greens)
    for i in range(5):
        currLet = guessOld == guessOld[i]
        greenLocs = np.logical_and(green,currLet)
        yellowLocs = np.logical_and(yellow,currLet)
        
        if sum(greenLocs) + sum(yellowLocs) != sum(currLet): #Condition is met; can place upper bound on number of letter repeats
            maxReps[i] = sum(greenLocs) + sum(yellowLocs)
            
            #If this condition is met, the signal from the repeat-associated grey letters has been accounted for. Set these to false
            currLetNG = np.logical_and(currLet,np.invert(greenLocs))
            currLetCumsum = np.cumsum(currLetNG)
            accountedGreys = np.logical_and(currLetCumsum > sum(yellowLocs),grey)
            grey[accountedGreys] = False
        else:
            maxReps[i] = 6
    
    #Check 1: Are any of the green letters not identical in the new guess to the old guess?
    if sum([guessOld[c] == guessNew[c] for c in range(5) if green[c]]) != sum(green):
        return False
    #Check 2: Are any of the yellow letters not in the new guess? Need to take into account repeated letters
    elif sum([sum(guessNew == guessOld[c]) > maxReps[c] for c in range(5)]) != 0: #Part 1: check letter limits aren't exceeded
        return False
    elif sum([guessOld[c] in guessNew for c in range(5) if yellow[c]]) != sum(yellow): #Part 2: check all yellows are accounted for
        return False
    elif sum([guessOld[c] == guessNew[c] for c in range(5) if yellow[c]]) != 0: #Part 3: check all yellows have different letters between the old and the new guesses
        return False
    #Check 3: Are any of the grey letters in the new guess?
    elif sum([guessOld[c] in guessNew for c in range(5) if grey[c]]) != 0:
        return False
    #If checks pass, this word is a permissible guess
    else:
        return True
    
#Function that provides the scoring of a given word
def scoreGuess(target,guess):
    green = target == guess
    
    #Make greens non-alphabetical values in the arrays so they are skipped in the next steps
    tgtNoGrn = target + green*50
    gssNoGrn = guess + green*100
    
    yellow = np.zeros((1,5),dtype='bool')
    for i in range(5):
        if ~green[i]:
            c = guess[i]

            #Double letters are a bit tricky - need to count the number of repeats in the target
            noCT = sum([d == c for d in tgtNoGrn]) #Total number of instances of c in target (ignoring greens)
            noCG = sum([gssNoGrn[j] == c for j in range(i)]) #Total number of instances of c in guess up to this point

            if noCT > noCG:
                yellow[0,i] = True
    
    return np.squeeze(yellow + 2*green) 
    
#Function that provides the ternary encoding of the complete set of targets with reference to a given guess
def scoreGuessTernary(targets,guess):
    guessAr = np.tile(guess,(np.size(targets,0),1))
    powAr = np.tile(np.array([1,3,9,27,81]),(np.size(targets,0),1))
    green = targets == guessAr
    ternTerms = np.multiply(powAr,green)*2 #Encode the green tiles
    
    #Make greens non-alphabetical values in the arrays so they are skipped in the next steps
    tgtNoGrn = targets + green*50
    gssNoGrn = guessAr + green*100
    
    
    for i in range(5): #Index of guess column
        gssColumn = gssNoGrn[:,i]
        jRange = np.arange(4)
        jRange[jRange>=i] += 1 
        
        tgtColumn = np.zeros((1,np.size(targets,0)),dtype='bool')
        for j in jRange: #Index of target column, skipping current i position
            tgtColumn = np.logical_or(tgtColumn,tgtNoGrn[:,j] == gssColumn)
            tgtNoGrn[tgtNoGrn[:,j] == gssColumn,j] += 150 #Prevent double counting of repeat letters
        
        if i == 0:
            yellow = tgtColumn.T
        else:
            yellow = np.concatenate((yellow,tgtColumn.T),axis=1)
        
    ternTerms += np.multiply(powAr,yellow) #Encode the yellow tiles
    
    return np.sum(ternTerms,axis=1)
    
#Function that finds the entropy of each guess given the target list
def scoreGuessEntropy(wordList,guess):
    ternSums = scoreGuessTernary(wordList,guess)
    
    [histCnt,binEdges] = np.histogram(ternSums,bins=(np.arange(pow(3,5))-0.5))
    binProbs = np.divide(histCnt,sum(histCnt))
    binProbsPos = binProbs #Create a copy of binProbs with non-zero entries to prevent numpy from generating warnings
    binProbsPos[binProbs == 0] += 0.0001
    
    return -sum(np.multiply(binProbs,np.nan_to_num(np.log2(binProbsPos))))

def scoreAllEntropies(letterStore):
    entropies = np.array([scoreGuessEntropy(letterStore,letterStore[i,:]) for i in range(np.size(letterStore,0))])
    
    return np.sort(entropies), np.argsort(entropies)

#Function that narrows down the possible wordlist based on a guess
def narrowWordList(guess,outcome,oldWordList):
    keepInds = [permitGuess(guess,outcome,oldWordList[c,:]) for c in range(np.size(oldWordList,0))]
    
    return oldWordList[keepInds,:]

def getLetterFrequencies(wordList):
    wordListFlat = np.reshape(wordList,[np.size(wordList),1])
    [histCnt,binEdges] = np.histogram(wordListFlat,bins=(np.arange(97,124)-0.5))
    
    return histCnt/sum(histCnt)

###### Automated benchmarking code ######

def runGuessCycleAuto(guess,tgt,letterStore,letterProbs):
    guessLine = word2line(guess)
    tgtLine = word2line(tgt)
    
    #Narrow down the word list based on guess
    letterStore = narrowWordList(guessLine,scoreGuess(tgtLine,guessLine),letterStore)
    
    #Find the best and worst-scoring candidates
    [entropyVals,entropyInds] = scoreAllEntropies(letterStore)
    
    #For all guesses that have equal entropy to the top choice, choose the one with the most common undetermined letters
    equalInds = entropyVals == entropyVals[-1]
    if sum(equalInds) == 1 or np.size(entropyVals) == 1:
        return letterStore, entropyInds[-1]
    else:
        firstInd = np.nonzero(np.diff(equalInds))
        if not firstInd[0]:
            firstInd = 0
        else:
            firstIndTmp = firstInd[0][0]
            firstInd = firstIndTmp + 1
        
        freqScores = np.zeros([np.size(entropyVals,0) - firstInd,1])
        thisScore = scoreGuess(tgtLine,guessLine)
        
        unknownLets = thisScore == 0
        
        for i in range(firstInd,np.size(entropyVals,0)):
            thisWord = letterStore[entropyInds[i],:]
            
            #Look up and add together the probabilities of the unknown letters
            freqScores[i-firstInd] = sum([letterProbs[thisWord[c]-97] for c in range(5) if unknownLets[c]])
        
        return letterStore, entropyInds[freqScores.argmax() + firstInd]
    
#Function that calculates how long the algorithm takes to guess a given target
def evalGuessingTime(tgt,letterStore,letterProbs):
    
    guessCnt = 1
    guess = 'tares' #The optimal initial guess
    
    while guess != tgt:
        guessCnt += 1
        
        letterStore, selectInd = runGuessCycleAuto(guess,tgt,letterStore,letterProbs)
        
        guess = line2word(letterStore[selectInd,:])
    
    return guessCnt

#Function that evaluates the guessing time for many random targets
def calcGuessTimeDist(noSamples):
    letterStoreInit = getData()
    tgtStore = getTgts()
    
    letterProbs = getLetterFrequencies(letterStoreInit)
    
    sampleList = np.random.choice(np.size(tgtStore,0), size=noSamples, replace=False)
    
    calcTimes = np.zeros((noSamples,1))
    for i in range(noSamples):
        currTgt = line2word(tgtStore[sampleList[i],:])
        letterStoreCpy = letterStoreInit
        calcTimes[i] = evalGuessingTime(currTgt,letterStoreCpy,letterProbs)
        
        print('Time was: ' + str(calcTimes[i]) + '. Percentage complete: ' + str(i*100/noSamples))
    
    return calcTimes

###### Manual IO code ######

#Function that runs a guess cycle, while outputing user-readable data for guess selection
def runGuessCycleManual(guess,outcome,letterStore):
    guessLine = word2line(guess)
    
    #Narrow down the word list based on guess
    letterStore = narrowWordList(guessLine,outcome,letterStore)
    
    #Find the best and worst-scoring candidates
    [entropyVals,entropyInds] = scoreAllEntropies(letterStore)
    
    summariseChoices(letterStore,entropyVals,entropyInds)
    
    return letterStore

#Function that summarises entropy statistics in user-readable fashion
def summariseChoices(letterStore,entropyVals,entropyInds):
    showDepth = min((10,np.size(letterStore,0)))
    print('Top ' + str(showDepth) + ' scoring candidates after this guess, with corresponding entropy:')
    for i in range(1,showDepth+1):
        print(line2word(letterStore[entropyInds[-i],:]) + ' | ' + str(entropyVals[-i]))

#Main manual mode program
def manualModeProgram():
    #Read in starting data
    letterStore = getData()
    
    #Get result 
    print('Optimal guess 1 is \'tares\'. Input result of this guess here (e.g. [0,1,2,0,0] using the encoding 0 = grey/white, 1 = yellow, 2 = green)')
    outcome1 = input('>>')
    validIn = validateScoreInput(outcome1)
    while not validIn:
        print('Please try inputting outcome of first guess again:')
        outcome1 = input('>>')
        validIn = validateScoreInput(outcome1)
    
    #Narrow down the word list based on first guess
    letterStore = runGuessCycleManual('tares',np.array(eval(outcome1)),letterStore)
    
    while True:
        print('Please input the word that you selected, or q to quit:')
        wordIn = input('>>')
        validIn = validateWordInput(wordIn)
        
        if wordIn == 'q':
            return None
        else:        
            while not validIn:
                print('Please try inputting your word choice again, or q to quit:')
                wordIn = input('>>')
                validIn = validateWordInput(wordIn)
                
                if wordIn == 'q':
                    return None
        
        print('And now input result of this guess here:')
        outcomeNew = input('>>')
        validIn = validateScoreInput(outcomeNew)
        while not validIn:
            print('Please try inputting outcome of this guess again:')
            outcomeNew = input('>>')
            validIn = validateScoreInput(outcomeNew)
        
        #Narrow down the word list based on new guess
        letterStore = runGuessCycleManual(wordIn,np.array(eval(outcomeNew)),letterStore)
        
#Function that validates the user input from 
def validateScoreInput(inStr):
    goodFlag = True
    
    #Check 1: Is input of the correct length?
    if len(inStr) != 11:
        goodFlag = False
    else:
        #Check 2: are brackets and commas in the right places?
        strFmt = '\[\d,\d,\d,\d,\d\]'
        if not re.findall(strFmt,inStr):
            goodFlag = False
        else:
            #Check 3: are all the digits between 0 and 2?
            numStore = np.array(eval(inStr))
            if sum(np.logical_and(numStore >=0, numStore < 3)) != 5:
                goodFlag = False
        
    return goodFlag

def validateWordInput(inStr):
    goodFlag = True
    
    #Check 1: Is input the correct length?
    if len(inStr) != 5:
        goodFlag = False
    else:    
        #Check 2: Is input between 97 and 122 when converted to ASCII code?
        ascForm = word2line(inStr)
        if sum(np.logical_and(ascForm > 96,ascForm < 123)) != 5:
            goodFlag = False
    
    return goodFlag