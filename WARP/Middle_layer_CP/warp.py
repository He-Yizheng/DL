import minizinc
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class WarpMiddleLayer:
    def __init__(self, R0, model_file=None):
        self.R0 = R0
        self.model_file = model_file if model_file else os.path.join(BASE_DIR, "warp.mzn")

    def solve(self):
        cp_solver = minizinc.Solver.lookup("com.google.ortools.sat")
        cp_inst = minizinc.Instance(solver=cp_solver, model=minizinc.Model())
        cp_inst.add_file(self.model_file)
        cp_inst["R0"] = self.R0

        start_time = time.time()
        result_all = cp_inst.solve(optimisation_level=2, all_solutions=True)

        end_time = time.time()
        print("FINISHED")
        print("Total time taken: %.2f seconds" % (end_time - start_time))

        solutions = []
        for result in result_all:
            x0 = []
            for i in range(32):
                for j in range(1, 16):
                    if result.x[0][i][j] == True:
                        x0.append([i, j])
            x = [[] for _ in range(32)]
            for i in range(32):
                for j in range(16):
                    if result.x[self.R0][i][j] == True:
                        x[i].append(j)
            solutions.append((x0, x))
        return solutions


if __name__ == "__main__":
    R = int(input("Input R: "))
    target = input("1: BSP, 2: PSP: ").strip()
    if target == "1":
        result_dir = os.path.join(BASE_DIR, "..", "DL_BSP", "Middle_layer", "result")
    else:
        result_dir = os.path.join(BASE_DIR, "..", "DL_PSP", "Middle_layer", "result")
    out_file = os.path.join(result_dir, "middle_layer_%d.txt" % R)

    solutions = WarpMiddleLayer(R).solve()
    with open(out_file, "w", encoding="utf-8") as file:
        file.write("SATISFIED\n")
        file.write("Time taken: ---\n")
        file.write("R0: %d\n" % R)
        for idx, (x0, x) in enumerate(solutions):
            file.write("\n\n--------------------------------------------------------------\n")
            file.write("Solution %d\n" % idx)
            file.write("x0 = %s\n" % x0)
            for i in range(32):
                file.write("x:%d \t%s\n" % (i, x[i]))
                file.write("--------------------------------------------------------------\n")
