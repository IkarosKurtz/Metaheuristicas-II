
"""
Qué tenemos que solucionar?? Hacemos lo del Obed en binario??
cual de obed?

Qué tenemos que encontrar un cromosoma que en binario dija "Obed Says Go Fuck Jusepe Fairon", el cromosoma
final deberia de ser:
"01001111011000100110010101100100001000000101001101100001011110010111001100100000010001110110111100100000011001100111010101100011011010110010000001001010011101010111001101100101011100000110010100100000010001100110000101101001011100100110111101101110"

acabo de bajar 2 kilos leo

Ya vez que queria cromosomas largos el Ornelas

Va, yo pensaba que era uno equis de que llegara a puros 1's
Pero si asi dijo el profe, asi lo hacemos

como quieres planearlo?

No se, que cosas tenemos que hacer??

- Mutación
- Ruleta
- Crossover




jaja

o sea, yo con lo que me quedé, o pensaba, era un algoritmo genético normal, de puros 1, lo que debíamos centrarnos era la selección
que fuera ruleta, lo que no sé, es cómo aplicar la ruleta en general, o sea no sé cómo funciona

Según yo es con lo del slot, dejame hago un ejemplo:

String  fitness   Probability  Slot
1       0.64      0.31         
2       0.32      0.15
3       0.68      0.33
4       0.4       0.19
Total   2.04     0.98 (deberia ser 1)

Ahora sí, basicamente es eso.
La ídea es que sacamos el fitness de todos los individuos y luego los sumamos
después calculamos la probabilidad de cada uno *fitness/total_fitness*,
deberia de dar *1*, pero por que quite decimales me dio `0.98`.
Lo importante es el slot, que es el que define si ese individuo se selecciona (no se para que se selecciona
eso nunca entiendo, no se si es para el crossover o que).

El slot se calcula con la probabilidad, *prob+prob_acumulada*, en el primero seria `0.32 + 0` entonces
la tabla seria:

String  fitness   Probability  Slot
1       0.64      0.31         0.31
2       0.32      0.15
3       0.68      0.33
4       0.4       0.19
Total   2.04     0.98 (deberia ser 1)

Y así vas con los demás. El siguiente es `0.15+0.31`

String  fitness   Probability  Slot
1       0.64      0.31         0.31
2       0.32      0.15         0.46
3       0.68      0.33         0.79
4       0.4       0.19         0.98 (deberia ser 1)
Total   2.04     0.98 (deberia ser 1)

Esa era la parte complicada. Sí se entendió??

https://cratecode.com/info/roulette-wheel-selection

encontre esa explicacion, pero no entendi bien


"""
from typing import Any

import numpy as np
from numpy.typing import NDArray

POPULATION_SIZE = 1500
TEMPLATE = "01001111 01100010 01100101 01100100 00100000 01010011 01100001 01111001 01110011 00100000 01000111 01101111 00100000 01100110 01110101 01100011 01101011 00100000 01001010 01110101 01110011 01100101 01110000 01100101 00100000 01000110 01100001 01101001 01110010 01101111 01101110 ".replace(
  " ", "")
CROMOSOME_SIZE = len(TEMPLATE)
MAX_GENERATIONS = 1000
MUTATION_PROB = 0.4
STOPPING_CRITERIA = 50


def generate_population():
  population = [
    [
      np.random.randint(0, 2)  # 2 es exclusivo
      for _ in range(CROMOSOME_SIZE)
    ]
    for _ in range(POPULATION_SIZE)
  ]

  return np.stack(population)


def fitness(individual: NDArray[Any], template: str) -> int:
  return np.sum(individual == np.array(list(map(int, template))))


def roulette_selection(population: NDArray[Any]):
  fitnesses = np.array([fitness(i, TEMPLATE) for i in population])
  total_fitness = sum(fitnesses)

  probabilities = fitnesses / total_fitness
  slots = np.cumsum(probabilities)

  random_numbers = np.random.rand(len(population))
  selected_individuals = population[np.searchsorted(slots, random_numbers)]

  return np.array(selected_individuals)


def crossover(father1: NDArray[Any], father2: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
  cross_point = np.random.randint(1, CROMOSOME_SIZE)

  son1 = np.concatenate((father1[:cross_point], father2[cross_point:]))
  son2 = np.concatenate((father2[:cross_point], father1[cross_point:]))

  return son1, son2


def mutation(individual: NDArray[Any]):
  for i in range(len(individual)):
    if np.random.rand() < MUTATION_PROB:
      individual[i] = 1 - individual[i]


def mutation(individuo: NDArray[Any]):
  for i in individuo:
    if np.random.rand() < MUTATION_PROB:
      individuo[i] = 1 - individuo[i]


# en teoria retorna los dos individuos con mayor fitness entre los dos hijos y padres
def replace(son1: NDArray[Any], son2: NDArray[Any], father1: NDArray[Any], father2: NDArray[Any]):
  array = [son1, son2, father1, father2]
  array_sorted = sorted(
    array,
    key=lambda ind: fitness(ind),
    reverse=True
  )

  return array_sorted[0], array_sorted[1]


def main():
  """
  Pasos:
  - Generar la población ✅
  - Calcular la fitness de cada individuo ✅
  - Seleccion por ruleta ✅
  - Cruzamiento
  - Mutacion
  - Repeat
  """

  """
  Tu eres el master, qué se hace en el cruzamiento, o como se hace el ciclo de las generaciones
  lo del cruzamiento pues es lo de combinar a los 2 padres, tons yo digo que hagamos el cruzamiento simple, no del OX o esos
  solo lo partimos a la mitad

  tipo
  p1 - 0101010101
  p2 - 0000110111

  partido a la mitad por decir algo

  h1 - 0101010111
  h2 - 0000110101

  bueno algo asi, namas combinamos y ya

  y pues lo de las generaciones es el loop para conseguir mejor respuesta
  
  Aguantame unos minutos

  va, mientras trato de ver qpd. 
  Ya ando
  """

  population = generate_population()
  best = 0
  best_individual = None
  iters_with_same_best = 0
  print("CROMOSOME_SIZE: ", CROMOSOME_SIZE)
  print("MUTATION_PROB: ", MUTATION_PROB)

  # CICLO??
  for _ in range(MAX_GENERATIONS):
    print(f"Generation: {_ + 1}")
    fitnesses = np.array([fitness(i, TEMPLATE) for i in population])
    new_best = fitnesses[fitnesses.argmax()]
    new_best_individual = population[fitnesses.argmax()]

    if new_best > best:
      best = new_best
      best_individual = new_best_individual
      print(f"New best: {new_best} - {np.array(new_best_individual)}")
      iters_with_same_best = 0
    elif new_best == best:
      iters_with_same_best += 1

    if iters_with_same_best == STOPPING_CRITERIA:
      print("Stopping because of no improvement")
      break

    best_selection = roulette_selection(population)
    new_population = []

    for i in range(0, len(best_selection), 2):
      son1, son2 = crossover(best_selection[i], best_selection[i + 1])
      mutation(son1)
      mutation(son2)

      new_population.append(son1)
      new_population.append(son2)

    population = np.array(new_population)

  print(f"\n\nBest: {best} - {np.array(best_individual)}")

  best_str = "".join(str(i) for i in best_individual)
  template_str = "".join(str(i) for i in TEMPLATE)
  print(f"\n\nBest individual:\n{best_str}")
  print(f"Template:\n{template_str}")
  print("Strings are: ", "Not Equals" if best_str != template_str else "Equals")


if __name__ == "__main__":
  main()
