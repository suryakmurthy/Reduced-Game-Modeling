import gurobipy as gp
from gurobipy import GRB

m = gp.Model("simple")

x = m.addVar(name="x")
y = m.addVar(name="y")

m.setObjective(x + y, GRB.MAXIMIZE)
m.addConstr(x + 2*y <= 4)

m.optimize()

for v in m.getVars():
    print(v.varName, v.x)

print("Objective:", m.objVal)