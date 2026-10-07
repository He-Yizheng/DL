import minizinc
import time

R = 6
cp_solver = minizinc.Solver.lookup("cp-sat")


cp_inst = minizinc.Instance(solver=cp_solver, model=minizinc.Model())
cp_inst.add_file("LBlock/Middle_layer/LBlock.mzn")
cp_inst["R"] = R
start_time = time.time()
result_all = cp_inst.solve(optimisation_level=2, all_solutions=True)

file = open("LBlock/Middle_layer/result_%d.txt" % R, "w")
file.write(str(result_all.status) + "\n")
end_time = time.time()
file.write("Time taken: %.2f seconds\n" % (end_time - start_time))
file.write("R0: %d\n" % R)

count = 0
for result in result_all:
    file.write("\n\n--------------------------------------------------------------\n")
    file.write("Solution %d\n" % count)
    x = []
    for i in range(8):
        for j in range(1, 16):
             if result.x[0][0][i][j] == True:
                 x.append([i, j])
    for i in range(8):
        for j in range(1, 16):
            if result.x[0][1][i][j] == True:
                x.append([i + 8, j])
    file.write("x0 = %s\n" % x)
    x = [[] for _ in range(16)]
    for i in range(8):
        for j in range(16):
            if result.x[R][0][i][j] == True:
                x[i].append(j)
    for i in range(8):
        for j in range(16):
            if result.x[R][1][i][j] == True:
                x[i + 8].append(j)
    for i in range(16):
        file.write("x:%d \t%s\n" % (i, x[i]))
        file.write("--------------------------------------------------------------\n")
    count += 1
file.close()
print("finished")
end_time = time.time()
print("Total time taken: %.2f seconds" % (end_time - start_time))
    