#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:02:11 2020

@author: philipwinchester
"""

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.chdir('/Users/philipwinchester/Documents/SoccermaticsForPython-master/Work/D_C/Python')
import Functions as funcs

Match_Data = funcs.LoadData("01/08/18")

Teams = sorted(list(set(Match_Data['HomeTeam'])))
Res = funcs.Optimise2(Match_Data, Teams)

t = time.time()
(real_madrid_position_counts, GoalsFor, GoalsAgainst, Losses, Wins, Draws) = funcs.simulate_n_seasons(Teams, Res, Res['Gamma'][0],  Res['Rho'][0])
elapsed = time.time() - t

"""nn = 1000
Firstlist = [0]*41
Winslist = [0]*41
Drawslist = [0]*41
Winslist = [0]*41
Losseslist = [0]*41

count = 0
for i in range(-20,21):
    print(i)
    (real_madrid_position_counts, GoalsFor, GoalsAgainst, Losses, Wins, Draws) = funcs.simulate_n_seasons(Teams, Res, Res['Gamma'][0],  Res['Rho'][0], RealMadridDefenceChange=i/38, n =nn)
    if str(type(real_madrid_position_counts.get(1))) == "<class 'NoneType'>":
        Firstlist[count]
    else:
        Firstlist[count] = real_madrid_position_counts.get(1)/nn

    Winslist[count] = Wins
    Losseslist[count] = Losses
    Drawslist[count] = Draws
    count += 1


d = {'Fewer Goals Let in': np.arange(-20,21), 'First Prop': Firstlist, 'Wins': Winslist, 'Losses': Losseslist, 'Draws': Drawslist}
fd = pd.DataFrame(data=d)
fd.to_csv('DefenceData.csv', index = False)


Def = pd.read_csv('DefenceData.csv')
At = pd.read_csv('AttackData.csv')


Defrow = Def.iloc[20][1:6]
Atrow = At.iloc[20][1:6]
m = (Defrow + Atrow)/2


Def.to_csv('DefenceData.csv', index = False)
At.to_csv('AttackData.csv', index = False)"""
