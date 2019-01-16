from projectq.ops import All, CNOT, H, Measure, X, Z, CZ
from projectq import MainEngine

quantumEngine = MainEngine()
q1 = quantumEngine.allocate_qubit()
q2 = quantumEngine.allocate_qubit()
q3 = quantumEngine.allocate_qubit()
q4 = quantumEngine.allocate_qubit()

print("Bob sending:")
print("1", "1")
# Below define what to send
X | q1
X | q2
Measure | q1
Measure | q2

# Perform the super dense coding algorithm
H | q3
CNOT | (q3, q4)
CZ | (q1, q3)
if int(q2) == 1:
    X | q3
CNOT | (q3, q4)
H | q3

# alice has recieved the bits, now lets measure them
Measure | q3
Measure | q4
print("Alice Recieved:")
print(int(q3), int(q4))