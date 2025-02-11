"""
  Algoritmo genético para resolver el problema de onemax, que consiste en
  encontrar un sujeto con puros 1's.
  
  Maestro: Francisco Javier Ornelas Zapata
  Materia: Metaheurísticas II
  Equipo: 
    Carlos Leonardo Cruz Ortiz
    Iván Israel Hurtado Lozano
    José Luis Elizondo Figueroa  
  
  Dependencias:
    - numpy
"""

from typing import Any
from rich.console import Console
from rich.pretty import pprint

import numpy as np
from numpy.typing import NDArray

POPULATION_SIZE = 1500
CROMOSOME_SIZE = 500
MAX_GENERATIONS = 2000
STOP_CRITERIA = 200
MUTATION_PROB = 1 / CROMOSOME_SIZE
NUMBER_OF_SELECTED_SOLUTIONS = 100


def generate_population() -> NDArray[Any]:
  population = [
    [
      np.random.randint(0, 2)  # 2 es exclusivo
      for _ in range(CROMOSOME_SIZE)
    ]
    for _ in range(POPULATION_SIZE)
  ]

  return np.stack(population)


def fitness(candidate_solution: NDArray[Any]) -> int:
  return np.sum(candidate_solution)


def roulette_selection(population: NDArray[Any]):
  fitnesses = np.array([fitness(i) for i in population])
  total_fitness = sum(fitnesses)

  probabilities = fitnesses / total_fitness
  slots = np.cumsum(probabilities)

  random_numbers = np.random.rand(NUMBER_OF_SELECTED_SOLUTIONS)
  roulette_winners = population[np.searchsorted(slots, random_numbers)]

  return np.array(roulette_winners)


def crossover(father1: NDArray[Any], father2: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
  cross_point = np.random.randint(1, CROMOSOME_SIZE)

  son1 = np.concatenate((father1[:cross_point], father2[cross_point:]))
  son2 = np.concatenate((father2[:cross_point], father1[cross_point:]))

  return son1, son2


def mutation(candidate_solution: NDArray[Any]):
  for i in range(len(candidate_solution)):
    if np.random.rand() < MUTATION_PROB:
      candidate_solution[i] = 1 - candidate_solution[i]


# en teoria retorna los dos individuos con mayor fitness entre los dos hijos y padres
def replace(son1: NDArray[Any], son2: NDArray[Any], father1: NDArray[Any], father2: NDArray[Any]):
  array = [son1, son2, father1, father2]
  array_sorted = sorted(
    array,
    key=lambda ind: fitness(ind),
    reverse=True
  )

  return array_sorted[0], array_sorted[1]


def better_print(solution: NDArray[Any], *, values: int = 4) -> list[Any]:
  return np.concatenate((solution[:values], ["..."], solution[-values:])).tolist()


def main():
  population = generate_population()
  console = Console()
  highest_fitness = 0
  optimal_solution = None
  repeted_best = 0
  console.print(f"[green][!] CROMOSOME_SIZE: {CROMOSOME_SIZE}")
  console.print(f"[green][!] MUTATION_PROB: {MUTATION_PROB * 100}%")
  console.print(f"[green][!] NUMBER_OF_SELECTED_SOLUTIONS: {NUMBER_OF_SELECTED_SOLUTIONS}")

  for generation in range(MAX_GENERATIONS):
    if generation % 100 == 0:
      console.print(f"[green][🔁] Generation: {generation + 1}\n")

    fitnesses = np.array([fitness(i) for i in population])
    new_highest_fitness = fitnesses[fitnesses.argmax()]
    new_optimal_solution = population[fitnesses.argmax()]

    if new_highest_fitness > highest_fitness:
      highest_fitness = new_highest_fitness
      optimal_solution = new_optimal_solution

      console.print(f"\n[red][🔴] New highest fitness: {new_highest_fitness}", end="\n")
      pprint(better_print(new_optimal_solution, values=5))

      repeted_best = 0
    elif repeted_best == STOP_CRITERIA:
      console.print("[yellow][💀] The optimal solution hasn't been improved for a long time")
      break
    else:
      repeted_best += 1

    if highest_fitness == CROMOSOME_SIZE:
      console.print("[green][🎉] Solution found!")
      break

    selected_candidates = roulette_selection(population)
    new_population = []

    while len(new_population) < POPULATION_SIZE:
      i, y = np.random.choice(len(selected_candidates), 2, replace=False)

      first_parent, second_parent = selected_candidates[i], selected_candidates[y]
      first_child, second_child = crossover(first_parent, second_parent)

      mutation(first_child)
      mutation(second_child)

      first_child, second_child = replace(first_child, second_child, first_parent, second_parent)

      new_population.append(first_child)
      new_population.append(second_child)

    population = np.stack(new_population)

  console.print(f"\n\n[green][🎉] Best: {highest_fitness}", end="\n")
  pprint(np.array(optimal_solution))

  best_str = "".join(str(i) for i in optimal_solution)
  console.print(f"\n\n[green][🎉] Best individual:", end="\n")
  console.print(best_str)


if __name__ == "__main__":
  main()
