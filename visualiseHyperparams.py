import matplotlib.pyplot as plt
import os
import numpy as np

params = {}
params['lr'] = [0, 1e-3, 2e-3, 1e-4, 2e-4, 1e-5, 2e-5, 1e-6, 2e-6]
params['wd'] = [0, 0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
params['d_model'] = [0, 16, 32, 64, 128, 256, 512]
params['n_heads'] = [0, 2,4,8,16]
params['n_layers'] = [0, 1,2,3,4]
params['d_ff'] = [0, 128,256,512,1024,2048]
params['batch_size']= [0, 16,32,64,128,256,512]
params['dropout'] = [0,0.1,0.2,0.3,0.4,0.5]
params['margin'] = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
params['epsilon'] = [0, 1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]

# Stap 1: kies een parameter en itereer over alle tuning modellen train bestanden om dit te vergaren
# Stap 2: itereer over alle mapscores voor MAPs
# Resultaat zijn twee lijsten van elk 200 groot
# Stap 3: Sommeer de MAP voor elke mogelijke waarde van nheads
# Resultaat is een lijst even groot als het aantal waardes die nheads aan kan nemen
# Stap 4: tel de frequentie van elke waarde van de parameter
# Stap 5: Deel de MAP waarde door de frequentie
# Stap 6: Plot de AVG waarbij x-as is waardes NHEADS en y-as is MAP score

param = 'dropout'
paramValues, mapValues, mapValuesPerParamValue = [], [], [[] for _ in range(len(params[param]))]

# Stap 1: vergaar param values
for i in range(1, 9):
    for file in os.listdir(f"Results/Experiments(mar2025)/Tuning/Model {i}"):
        if file.split('.')[0][-5:] != 'Train':
            continue
        with open(f"Results/Experiments(mar2025)/Tuning/Model {i}/{file}", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(':')
                if line[0] != param:
                    continue
                else:
                    paramValues.append(float(line[1]))
                    break

# Stap 2: vergaar MAP waardes
for i in range(1, 9):
    for file in os.listdir(f"Results/MAP scores/Tuning/Model {i}"):
        with open(f"Results/MAP scores/Tuning/Model {i}/{file}", 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip('\n')
            #print(line.split(' '))
            mapValues.append(float(line.split(' ')[-2]))
    
# Stap 3 en 4: Sommeer de MAP waardes voor elke mogelijke nhead parameter waarde en tel hoe vaak ze voorkomen

#sumMaps, count = np.zeros(len(params[param])), np.zeros(len(params[param]))


for i in range(len(mapValues)):
    for j in range(len(params[param])):
        if round(paramValues[i],1) == params[param][j]:
            mapValuesPerParamValue[j].append(mapValues[i])

# Stap 5: neem AVG over de gesommeerde mAPS
#sumMaps /= count
# Stap 6: plot de parameter tegen de MAP waardes

#Y = sumMaps
#print(sumMaps)
plt.boxplot(mapValuesPerParamValue[1:], showmeans=True)
ax = plt.gca()
ax.set_ylim([0.04, 0.28])
plt.xticks(range(0, len(mapValuesPerParamValue)), labels=params[param])
plt.grid(True, linestyle='--')
plt.xlabel(f"Dropout")
plt.ylabel("MAP score")
plt.title(f"Influence dropout")
plt.legend()
plt.show()