op1 = input()
operacio = str(input())
op2 = input()

correcte = True

if operacio not in "+-*:":
    raise SyntaxError("ERROR: Operació no definida")

try:
    op1 = float(op1)
    op2 = float(op2)
except:
    raise ValueError("ERROR: Operands han de ser nombres")

try:
    op1 / op2
except:
    raise ZeroDivisionError("ERROR: Divisió per zero")

if operacio == "+":
    res = op1 + op2
elif operacio == "-":
    res = op1 - op2
elif operacio == "*":
    res = op1 * op2
elif operacio == ":":
    res = op1 / op2

print(op1, operacio, op2, "=", float(res))