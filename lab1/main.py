import numpy as np

#Zadanie 2

plik1 = np.loadtxt('diabetes.txt', dtype=str)
plik2 = np.loadtxt('diabetes-type.txt', dtype=str)

#Zadanie 3a

klasy = np.unique(plik1[:, -1])

print("Symbole klas decyzyjnych:")
for c in klasy:
    print(c)

print("------------------------")

#Zadanie 3b

klasy2 = np.unique(plik1[:, -1], return_counts=True)

print("Wielkości klas decyzyjnych:")
for c in klasy2:
    print(c)

print("------------------------")

#Zadanie 3c

print("Minimalne i maksymalne wartości poszczególnych atrybutów")

plik3 = np.loadtxt('diabetes.txt')

for i in range(plik3.shape[1]):
    min_val = np.min(plik3[:, i])
    max_val = np.max(plik3[:, i])
    print("Atrybut", i+1, "- minimalna wartość:", min_val, "maksymalna wartość:", max_val)

print("------------------------")

#Zadanie 3d

for i in range(plik3.shape[1]):
    unique_values, counts = np.unique(plik3[:, i], return_counts=True)
    num_unique_values = len(unique_values)
    print("Atrybut", i+1, "- liczba unikalnych wartości:", num_unique_values)

print("------------------------")

#Zadanie 3e

for i in range(plik3.shape[1]):
    unique_values = np.unique(plik3[:, i])
    print("Atrybut", i+1, "- unikalne wartości:", unique_values)

print("------------------------")

#Zadanie 3f

std_total = np.std(plik3[:, :-1], axis=0)
print("Odchylenie standardowe dla każdego atrybutu w całym systemie:")
print(std_total)

#Zadanie 4

plik4 = np.loadtxt('diabetes.txt', dtype='object')

missing_mask = np.random.choice([False, True], size=plik4.shape, p=[0.9, 0.1])
plik4[missing_mask] = '?'
data_numeric = np.where(plik4 == '?', np.nan, plik4.astype(float))

means = np.nanmean(data_numeric, axis=0)
missing_values = np.isnan(data_numeric)
data_numeric[missing_values] = np.take(means, np.where(missing_values)[1])

modes = np.array([np.nan] * plik4.shape[1])
for i in range(plik4.shape[1]):
    if not np.issubdtype(data_numeric[:, i].dtype, np.number):
        modes[i] = np.nanmax(np.unique(plik4[:, i], return_counts=True), axis=1)[0]
        missing_values = plik4[:, i] == '?'
        plik4[missing_values, i] = modes[i]

print(plik4)
