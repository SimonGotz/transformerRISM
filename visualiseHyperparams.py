import matplotlib.pyplot as plt
import os

params = {}
params['n_heads'] = [2,4,8,16]

# Stap 1: kies een parameter en itereer over alle tuning modellen train bestanden om dit te vergaren
# Stap 2: itereer over alle mapscores voor MAPs
# Resultaat zijn twee lijsten van elk 200 groot
# Stap 3: Sommeer de MAP voor elke mogelijke waarde van nheads
# Resultaat is een lijst even groot als het aantal waardes die nheads aan kan nemen
# Stap 4: tel de frequentie van elke waarde van de parameter
# Stap 5: Deel de MAP waarde door de frequentie
# Stap 6: Plot de AVG waarbij x-as is waardes NHEADS en y-as is MAP score

param = 'n_heads'
paramValues, mapValues = [], []

# Stap 1: vergaar param values
# TODO VERANDER RANGE NAAR 8 ALS TUNING KLAAR IS
for i in range(1, 8):
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
                    paramValues.append(int(line[1]))
                    break

# Stap 2: vergaar MAP waardes
for i in range(1, 8):
    for file in os.listdir(f"Results/MAP scores/Tuning/Model {i}"):
        with open(f"Results/MAP scores/Tuning/Model {i}/{file}", 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip('\n')
            #print(line.split(' '))
            mapValues.append(float(line.split(' ')[-2]))
    
# Stap 3 en 4: Sommeer de MAP waardes voor elke mogelijke nhead parameter waarde en tel hoe vaak ze voorkomen

sumMaps, count = [0,0,0,0], [0,0,0,0]

for i in range(len(mapValues)):
    if paramValues[i] == 2:
        count[0] += 1
        sumMaps[0] += mapValues[i]
    elif paramValues[i] == 4:
        count[1] += 1
        sumMaps[1] += mapValues[i]
    elif paramValues[i] == 8:
        count[2] += 1
        sumMaps[2] += mapValues[i]
    elif paramValues[i] == 16:
        count[3] += 1
        sumMaps[3] += mapValues[i]

# Stap 5: neem AVG over de gesommeerde mAPS
for i in range(len(sumMaps)):
    sumMaps[i] /= count[i]

# Stap 6: plot de parameter tegen de MAP waardes

X = params['n_heads']
Y = sumMaps
plt.plot(range(len(sumMaps)),Y,'bo')
ax = plt.gca()
ax.set_ylim([0.11, 0.16])
plt.xticks(range(len(sumMaps)), labels=X)
plt.grid(True, linestyle='--')
plt.xlabel("Number of heads")
plt.ylabel("Average MAP score")
plt.title("Influence number of heads")
plt.show()