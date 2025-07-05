base = float(input())

anys = int(input())

if anys < 3:
    final = base*1.01
elif anys < 5:
    final = base*1.02
else:
    final = base*1.035

print("El salari final és: " + str(final))