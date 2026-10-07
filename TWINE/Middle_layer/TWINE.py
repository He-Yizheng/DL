import minizinc
import time

R = 4
cp_solver = minizinc.Solver.lookup("cp-sat")


cp_inst = minizinc.Instance(solver=cp_solver, model=minizinc.Model())
cp_inst.add_file("TWINE/Middle_layer/TWINE.mzn")
cp_inst["R"] = R
start_time = time.time()
result_all = cp_inst.solve(optimisation_level=2, all_solutions=True)

file = open("TWINE/Middle_layer/result_%d.txt" % R, "w")
file.write(str(result_all.status) + "\n")
end_time = time.time()
file.write("Time taken: %.2f seconds\n" % (end_time - start_time))
file.write("R: %d\n" % R)

count = 0
for result in result_all:
    file.write("\n\n--------------------------------------------------------------\n")
    file.write("Solution %d\n" % count)
    x0 = []
    for i in range(16):
        for j in range(1, 16):
            if result.x[0][i][j] == True:
                x0.append([i, j])
    file.write("x0 = %s\n" % x0)
    x = [[] for _ in range(32)]
    for i in range(16):
        for j in range(16):
            if result.x[R][i][j] == True:
                x[i].append(j)           
    for i in range(16):
        file.write("x:%d \t%s\n" % (i, x[i]))
        file.write("--------------------------------------------------------------\n")
    count += 1
file.close()
print("finished")
end_time = time.time()
print("Total time taken: %.2f seconds" % (end_time - start_time))
    