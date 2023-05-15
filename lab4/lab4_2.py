#Zadanie 1

system_decyzyjny = [
  [1, 1, 1, 1, 3, 1, 1],
  [1, 1, 1, 1, 3, 2, 1],
  [1, 1, 1, 3, 2, 1, 0],
  [1, 1, 1, 3, 3, 2, 1],
  [1, 1, 2, 1, 2, 1, 0],
  [1, 1, 2, 1, 2, 2, 1],
  [1, 1, 2, 2, 3, 1, 0],
  [1, 1, 2, 2, 4, 1, 1],
]

def znajdz_regule(system_decyzyjny):
  for wiersz in system_decyzyjny:
    decyzja = wiersz[-1]
    warunki = wiersz[:-1]
    pasuje = True
    for i in range(len(warunki)):
      if warunki[i] != 'x' and warunki[i] != warunki[i-1]:
        pasuje = False
        break
    if pasuje:
      return (warunki, decyzja)
  return None

regula = znajdz_regule(system_decyzyjny)
if regula is not None:
  print("Znaleziona reguła: warunki = ", regula[0], ", decyzja = ", regula[1])
else:
  print("Nie znaleziono reguły")

