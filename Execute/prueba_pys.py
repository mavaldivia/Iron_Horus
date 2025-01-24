import pyscipopt as pys

# Crear el modelo SCIP
model_pys = pys.Model("MILP_from_LP")


# Leer el archivo LP
model_pys.readProblem("modelo_0.lp")
model_pys.setParam("display/verblevel", 5)  # Nivel de detalle máximo


# Definir que es un problema de maximización
model_pys.setMaximize()

print('Optimizar')
# Optimizar el problema
model_pys.optimize()

# Obtener el estado de la solución
status = model_pys.getStatus()
print(f"Estado del modelo: {status}")

# Si se encuentra una solución, imprimir los resultados
if status == "optimal" or status == "feasible":
    print("Valor de la función objetivo:", model_pys.getObjVal())
    print("Valores de las variables:")
    # for var in model.getVars():
    #    print(f"{var.name}: {model.getVal(var)}")
else:
    print("No se encontró una solución.")