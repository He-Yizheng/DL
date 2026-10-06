import minizinc
import time

R = 7
cp_solver = minizinc.Solver.lookup("com.google.ortools.sat")


cp_inst = minizinc.Instance(solver=cp_solver, model=minizinc.Model())
cp_inst.add_file("DL/SKINNY/Middle_layer/Truncated/trail.mzn")
cp_inst["R"] = R
start_time = time.time()
result = cp_inst.solve(optimisation_level=2)

file = open("DL/SKINNY/Middle_layer/Truncated/result/trail_%d.txt" % R, "w")
file.write(str(result.status) + "\n")
end_time = time.time()
file.write("Time taken: %.2f seconds\n" % (end_time - start_time))
file.write("R: %d\n" % R)

x = [[] for _ in range(R + 1)]
for r in range(R + 1):
    for i in range(16):
        if result["x"][r][i] == True:
            x[r].append(i)
file.write("X:\n")
for r in range(R + 1):
    file.write("R%d:\t%s\n" % (R - r, x[r]))
    
y = [[] for _ in range(R + 1)]
for r in range(R + 1):
    for i in range(16):
        if result["y"][r][i] == True:
            y[r].append(i)
file.write("Y:\n")
for r in range(R, -1, -1):
    file.write("R%d:\t%s\n" % (R - r, y[r]))

c = [[] for _ in range(R + 1)]
for r in range(R + 1):
    for i in x[r]:
        if i in y[R - r]:
            c[r].append(i)
file.write("C:\n")
for r in range(R  + 1):
    file.write("C%d:\t%s\n" % (r, c[r]))
    
file.write("\n%s" % c)

file.close()

print(x)

print(y)

print(c)

print("finished")