import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from rich.console import Console
from rich.table import Table

console = Console()


class EnergyDistributionProblem(Problem):
  def __init__(self):
    super().__init__(
        n_var=4,
        n_obj=4,
        xl=[0.0, 0.0, 0.0, 0.0],
        xu=[1.0, 100.0, 50.0, 1.0]
    )

  def _evaluate(self, X, out, *args, **kwargs):
    # Extraemos cada variable
    x0 = X[:, 0]  # Fracción de energía renovable
    x1 = X[:, 1]  # Inversión en infraestructura
    x2 = X[:, 2]  # Nivel de mantenimiento
    x3 = X[:, 3]  # Factor de eficiencia operativa

    # Objetivo 1: Minimizar las pérdidas de energía. Se modela como una función de
    # (1 - x0)² (menor renovabilidad genera mayores pérdidas) y la desviación de la inversión óptima (50)
    # Además se penaliza una baja inversión en mantenimiento (ideal cerca de 50).
    f1 = (1 - x0)**2 + ((x1 - 50)**2) / 1000 + ((50 - x2)**2) / 1000

    # Objetivo 2: Maximizar la utilización de fuentes renovables. Se busca maximizar x0 y también se
    # considera la eficiencia operativa, por ello se define como negativo de una combinación lineal.
    f2 = - (0.7 * x0 + 0.3 * x3)

    # Objetivo 3: Minimizar los costos operativos. Se consideran los costos por inversión (cuadrático),
    # el costo adicional de no usar renovables y el costo asociado al mantenimiento.
    f3 = (x1**2) / 100 + (1 - x0) * 10 + x2 * 5

    # Objetivo 4: Maximizar la confiabilidad del sistema. Se modela combinando la calidad de las fuentes
    # renovables, la eficiencia operativa y el efecto del mantenimiento. Se toma el negativo para maximizar.
    f4 = - (x0 + x3 + x2 / 50)

    out["F"] = np.column_stack([f1, f2, f3, f4])


problem = EnergyDistributionProblem()

# Generamos direcciones de referencia para 4 objetivos. Aquí se usa el método "das-dennis" con n_partitions=5
ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=4)

algorithm = MOEAD(
  ref_dirs,
  n_neighbors=15,
  prob_neighbor_mating=0.7
)

res = minimize(
  problem,
  algorithm,
  ('n_gen', 200),
  seed=12,
  verbose=False
)

# Impresión de resultados: soluciones no dominadas obtenidas
console.print("Soluciones no dominadas:")
for i, sol in enumerate(res.F):
  console.print(f"\nSolución {i+1}:")
  console.print(f"  Pérdidas de energía: {sol[0]:.4f}")
  console.print(f"  Utilización de renovables: {-sol[1]:.4f}")
  console.print(f"  Costos operativos: {sol[2]:.4f}")
  console.print(f"  Confiabilidad: {-sol[3]:.4f}")

# Creación de una tabla para mostrar los resultados
table = Table(title="Soluciones No Dominadas", header_style="bold")

# Agregar columnas a la tabla
table.add_column("Solución", justify="center", style="cyan", no_wrap=True)
table.add_column("Pérdidas de Energía", justify="center", style="magenta")
table.add_column("Utilización de Renovables", justify="center", style="green")
table.add_column("Costos Operativos", justify="center", style="yellow")
table.add_column("Confiabilidad", justify="center", style="blue")

# Agregar filas con los resultados
for i, sol in enumerate(res.F):
  table.add_row(
    f"{i+1}",
    f"{sol[0]:.4f}",
    f"{-sol[1]:.4f}",
    f"{sol[2]:.4f}",
    f"{-sol[3]:.4f}"
  )

# Mostrar la tabla en la consola
console.print(table)
