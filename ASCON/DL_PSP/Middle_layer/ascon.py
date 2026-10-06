import minizinc
import time
import os

try:
    import resource 
except ImportError:
    resource = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_memory_limit(max_gb):
    if resource is None:
        return
    try:
        limit = max_gb * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError) as e:
        print("%s" % e)

class AsconMiddleLayer:
    def __init__(self, R, model_file=None, memory_limit_gb=128):
        self.R = R
        self.memory_limit_gb = memory_limit_gb
        self.file = model_file if model_file else os.path.join(BASE_DIR, "temp.mzn")
        set_memory_limit(memory_limit_gb)
        self.mzn = r"""
include "xor.mzn";


int: R;

array[0..R, 0..63, 0..31] of var bool: x;
array[0..(R - 1), 0..63, 0..31] of var bool: x_sb;
array[0..(R - 1), 0..63, 0..4, 0..1] of var bool: x_xor;


predicate sbox_layer(array[0..31] of var bool: a, array[0..31] of var bool: b) = 
    b[0] = (a[0])
    /\
    b[1] = (a[3] \/ a[6] \/ a[9] \/ a[10] \/ a[12] \/ a[13] \/ a[14] \/ a[18] \/ a[26] \/ a[28])
    /\
    b[2] = (a[7] \/ a[10] \/ a[11] \/ a[14] \/ a[19] \/ a[23] \/ a[26] \/ a[27] \/ a[31])
    /\
    b[3] = (a[6] \/ a[7] \/ a[9] \/ a[11] \/ a[13] \/ a[18] \/ a[25] \/ a[27] \/ a[28] \/ a[29] \/ a[31])
    /\
    b[4] = (a[9] \/ a[10] \/ a[14] \/ a[19] \/ a[20] \/ a[23] \/ a[24] \/ a[27] \/ a[31])
    /\
    b[5] = (a[3] \/ a[6] \/ a[13] \/ a[18] \/ a[20] \/ a[21] \/ a[24] \/ a[26] \/ a[27] \/ a[29] \/ a[31])
    /\
    b[6] = (a[4] \/ a[7] \/ a[8] \/ a[9] \/ a[11] \/ a[20] \/ a[24] \/ a[25] \/ a[26])
    /\
    b[7] = (a[6] \/ a[7] \/ a[8] \/ a[10] \/ a[11] \/ a[13] \/ a[14] \/ a[18] \/ a[20] \/ a[21] \/ a[24])
    /\
    b[8] = (a[9] \/ a[12] \/ a[13] \/ a[15] \/ a[25] \/ a[26] \/ a[28] \/ a[29] \/ a[30])
    /\
    b[9] = (a[1] \/ a[3] \/ a[6] \/ a[10] \/ a[15] \/ a[16] \/ a[18] \/ a[21] \/ a[30])
    /\
    b[10] = (a[7] \/ a[9] \/ a[10] \/ a[11] \/ a[13] \/ a[19] \/ a[23] \/ a[27] \/ a[28] \/ a[30])
    /\
    b[11] = (a[1] \/ a[6] \/ a[7] \/ a[11] \/ a[16] \/ a[18] \/ a[21] \/ a[26] \/ a[27] \/ a[30])
    /\
    b[12] = (a[10] \/ a[13] \/ a[15] \/ a[19] \/ a[20] \/ a[23] \/ a[24] \/ a[26] \/ a[27] \/ a[30])
    /\
    b[13] = (a[1] \/ a[3] \/ a[6] \/ a[9] \/ a[15] \/ a[18] \/ a[20] \/ a[24] \/ a[25] \/ a[27] \/ a[30])
    /\
    b[14] = (a[4] \/ a[7] \/ a[8] \/ a[11] \/ a[13] \/ a[20] \/ a[24] \/ a[29] \/ a[30])
    /\
    b[15] = (a[1] \/ a[6] \/ a[7] \/ a[8] \/ a[9] \/ a[10] \/ a[11] \/ a[18] \/ a[20] \/ a[24] \/ a[26] \/ a[30])
    /\
    b[16] = (a[3] \/ a[9] \/ a[12] \/ a[13] \/ a[18] \/ a[22] \/ a[25] \/ a[28] \/ a[29])
    /\
    b[17] = (a[2] \/ a[5] \/ a[6] \/ a[10] \/ a[14] \/ a[17] \/ a[21] \/ a[22] \/ a[26])
    /\
    b[18] = (a[9] \/ a[10] \/ a[11] \/ a[13] \/ a[14] \/ a[18] \/ a[22] \/ a[23] \/ a[26] \/ a[27] \/ a[28] \/ a[31])
    /\
    b[19] = (a[2] \/ a[5] \/ a[6] \/ a[11] \/ a[17] \/ a[21] \/ a[22] \/ a[27] \/ a[31])
    /\
    b[20] = (a[3] \/ a[5] \/ a[10] \/ a[13] \/ a[14] \/ a[18] \/ a[22] \/ a[23] \/ a[24] \/ a[27] \/ a[31])
    /\
    b[21] = (a[2] \/ a[6] \/ a[9] \/ a[17] \/ a[22] \/ a[24] \/ a[25] \/ a[26] \/ a[27] \/ a[31])
    /\
    b[22] = (a[4] \/ a[5] \/ a[8] \/ a[11] \/ a[13] \/ a[18] \/ a[22] \/ a[24] \/ a[26] \/ a[29])
    /\
    b[23] = (a[2] \/ a[6] \/ a[8] \/ a[9] \/ a[10] \/ a[11] \/ a[14] \/ a[17] \/ a[22] \/ a[24])
    /\
    b[24] = (a[1] \/ a[3] \/ a[5] \/ a[15] \/ a[16] \/ a[18] \/ a[22] \/ a[26] \/ a[30])
    /\
    b[25] = (a[2] \/ a[6] \/ a[9] \/ a[10] \/ a[12] \/ a[13] \/ a[15] \/ a[22] \/ a[28] \/ a[30])
    /\
    b[26] = (a[1] \/ a[5] \/ a[10] \/ a[11] \/ a[16] \/ a[18] \/ a[22] \/ a[23] \/ a[27] \/ a[30])
    /\
    b[27] = (a[2] \/ a[6] \/ a[9] \/ a[11] \/ a[13] \/ a[22] \/ a[25] \/ a[26] \/ a[27] \/ a[28] \/ a[29] \/ a[30])
    /\
    b[28] = (a[1] \/ a[3] \/ a[9] \/ a[10] \/ a[15] \/ a[18] \/ a[22] \/ a[23] \/ a[24] \/ a[26] \/ a[27] \/ a[30])
    /\
    b[29] = (a[2] \/ a[5] \/ a[6] \/ a[13] \/ a[15] \/ a[21] \/ a[22] \/ a[24] \/ a[27] \/ a[29] \/ a[30])
    /\
    b[30] = (a[1] \/ a[4] \/ a[8] \/ a[9] \/ a[11] \/ a[18] \/ a[22] \/ a[24] \/ a[25] \/ a[30])
    /\
    b[31] = (a[2] \/ a[5] \/ a[6] \/ a[8] \/ a[10] \/ a[11] \/ a[13] \/ a[21] \/ a[22] \/ a[24] \/ a[26] \/ a[30]);


constraint forall(r in 0..(R - 1), l in 0..63)(
    sbox_layer(
        array1d(0..31, [x[r, l, i] | i in 0..31]),
        array1d(0..31, [x_sb[r, l, i] | i in 0..31])
    )
);

constraint forall(r in 0..(R - 1), l in 0..63)(
    x_xor[r, l, 0, 0] = (x_sb[r, l, 0] \/ x_sb[r, l, 1] \/ x_sb[r, l, 2] \/ x_sb[r, l, 3] \/ x_sb[r, l, 4] \/ x_sb[r, l, 5] \/ x_sb[r, l, 6] \/ x_sb[r, l, 7] \/ x_sb[r, l, 8] \/ x_sb[r, l, 9] \/ x_sb[r, l, 10] \/ x_sb[r, l, 11] \/ x_sb[r, l, 12] \/ x_sb[r, l, 13] \/ x_sb[r, l, 14] \/ x_sb[r, l, 15])
    /\
    x_xor[r, l, 0, 1] = (x_sb[r, l, 16] \/ x_sb[r, l, 17] \/ x_sb[r, l, 18] \/ x_sb[r, l, 19] \/ x_sb[r, l, 20] \/ x_sb[r, l, 21] \/ x_sb[r, l, 22] \/ x_sb[r, l, 23] \/ x_sb[r, l, 24] \/ x_sb[r, l, 25] \/ x_sb[r, l, 26] \/ x_sb[r, l, 27] \/ x_sb[r, l, 28] \/ x_sb[r, l, 29] \/ x_sb[r, l, 30] \/ x_sb[r, l, 31])
    /\
    x_xor[r, l, 1, 0] = (x_sb[r, l, 0] \/ x_sb[r, l, 1] \/ x_sb[r, l, 2] \/ x_sb[r, l, 3] \/ x_sb[r, l, 4] \/ x_sb[r, l, 5] \/ x_sb[r, l, 6] \/ x_sb[r, l, 7] \/ x_sb[r, l, 16] \/ x_sb[r, l, 17] \/ x_sb[r, l, 18] \/ x_sb[r, l, 19] \/ x_sb[r, l, 20] \/ x_sb[r, l, 21] \/ x_sb[r, l, 22] \/ x_sb[r, l, 23])
    /\
    x_xor[r, l, 1, 1] = (x_sb[r, l, 8] \/ x_sb[r, l, 9] \/ x_sb[r, l, 10] \/ x_sb[r, l, 11] \/ x_sb[r, l, 12] \/ x_sb[r, l, 13] \/ x_sb[r, l, 14] \/ x_sb[r, l, 15] \/ x_sb[r, l, 24] \/ x_sb[r, l, 25] \/ x_sb[r, l, 26] \/ x_sb[r, l, 27] \/ x_sb[r, l, 28] \/ x_sb[r, l, 29] \/ x_sb[r, l, 30] \/ x_sb[r, l, 31])
    /\
    x_xor[r, l, 2, 0] = (x_sb[r, l, 0] \/ x_sb[r, l, 1] \/ x_sb[r, l, 2] \/ x_sb[r, l, 3] \/ x_sb[r, l, 8] \/ x_sb[r, l, 9] \/ x_sb[r, l, 10] \/ x_sb[r, l, 11] \/ x_sb[r, l, 16] \/ x_sb[r, l, 17] \/ x_sb[r, l, 18] \/ x_sb[r, l, 19] \/ x_sb[r, l, 24] \/ x_sb[r, l, 25] \/ x_sb[r, l, 26] \/ x_sb[r, l, 27])
    /\
    x_xor[r, l, 2, 1] = (x_sb[r, l, 4] \/ x_sb[r, l, 5] \/ x_sb[r, l, 6] \/ x_sb[r, l, 7] \/ x_sb[r, l, 12] \/ x_sb[r, l, 13] \/ x_sb[r, l, 14] \/ x_sb[r, l, 15] \/ x_sb[r, l, 20] \/ x_sb[r, l, 21] \/ x_sb[r, l, 22] \/ x_sb[r, l, 23] \/ x_sb[r, l, 28] \/ x_sb[r, l, 29] \/ x_sb[r, l, 30] \/ x_sb[r, l, 31])
    /\
    x_xor[r, l, 3, 0] = (x_sb[r, l, 0] \/ x_sb[r, l, 1] \/ x_sb[r, l, 4] \/ x_sb[r, l, 5] \/ x_sb[r, l, 8] \/ x_sb[r, l, 9] \/ x_sb[r, l, 12] \/ x_sb[r, l, 13] \/ x_sb[r, l, 16] \/ x_sb[r, l, 17] \/ x_sb[r, l, 20] \/ x_sb[r, l, 21] \/ x_sb[r, l, 24] \/ x_sb[r, l, 25] \/ x_sb[r, l, 28] \/ x_sb[r, l, 29])
    /\
    x_xor[r, l, 3, 1] = (x_sb[r, l, 2] \/ x_sb[r, l, 3] \/ x_sb[r, l, 6] \/ x_sb[r, l, 7] \/ x_sb[r, l, 10] \/ x_sb[r, l, 11] \/ x_sb[r, l, 14] \/ x_sb[r, l, 15] \/ x_sb[r, l, 18] \/ x_sb[r, l, 19] \/ x_sb[r, l, 22] \/ x_sb[r, l, 23] \/ x_sb[r, l, 26] \/ x_sb[r, l, 27] \/ x_sb[r, l, 30] \/ x_sb[r, l, 31])
    /\
    x_xor[r, l, 4, 0] = (x_sb[r, l, 0] \/ x_sb[r, l, 2] \/ x_sb[r, l, 4] \/ x_sb[r, l, 6] \/ x_sb[r, l, 8] \/ x_sb[r, l, 10] \/ x_sb[r, l, 12] \/ x_sb[r, l, 14] \/ x_sb[r, l, 16] \/ x_sb[r, l, 18] \/ x_sb[r, l, 20] \/ x_sb[r, l, 22] \/ x_sb[r, l, 24] \/ x_sb[r, l, 26] \/ x_sb[r, l, 28] \/ x_sb[r, l, 30])
    /\
    x_xor[r, l, 4, 1] = (x_sb[r, l, 1] \/ x_sb[r, l, 3] \/ x_sb[r, l, 5] \/ x_sb[r, l, 7] \/ x_sb[r, l, 9] \/ x_sb[r, l, 11] \/ x_sb[r, l, 13] \/ x_sb[r, l, 15] \/ x_sb[r, l, 17] \/ x_sb[r, l, 19] \/ x_sb[r, l, 21] \/ x_sb[r, l, 23] \/ x_sb[r, l, 25] \/ x_sb[r, l, 27] \/ x_sb[r, l, 29] \/ x_sb[r, l, 31])
);

constraint forall(r in 0..(R - 1), l in 0..63)(
    xor_operation(
        array1d(0..31, [x_sb[r, l, i] | i in 0..31]),
        array1d(0..1, [x_xor[r, (l + 45) mod 64, 0, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 36) mod 64, 0, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 3) mod 64, 1, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 25) mod 64, 1, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 63) mod 64, 2, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 58) mod 64, 2, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 54) mod 64, 3, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 47) mod 64, 3, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 57) mod 64, 4, i] | i in 0..1]),
        array1d(0..1, [x_xor[r, (l + 23) mod 64, 4, i] | i in 0..1]),
        array1d(0..31, [x[r + 1, l, i] | i in 0..31])
    )
);

constraint forall(l in 0..63)(
    sum(b in 0..31)(bool2int(x[0, l, b])) = 1
);
"""
        self.model_last = r"""
constraint sum(l in 0..63, b in 0..31)(bool2int(x[R, l, b])) < 2048;

solve satisfy;
"""
        

    def solve(self, input):
        init_constraints = "constraint forall(l in 0..63)(\n"
        for i in input:
            if i == input[0]:
                init_constraints += f"\tif l == {i[0]} then x[0, l, {i[1]}] = true\n"
            else:
                init_constraints += f"\telseif l == {i[0]} then x[0, l, {i[1]}] = true\n"
        init_constraints += "\telse x[0, l, 0] = true\n\tendif\n"
        init_constraints += ");\n"

        file = open(self.file, "w", encoding="utf-8")
        file.write(self.mzn)
        file.write(init_constraints)
        file.write(self.model_last)
        file.close()

        cp_solver = minizinc.Solver.lookup("com.google.ortools.sat")
        cp_inst = minizinc.Instance(solver=cp_solver, model=minizinc.Model())
        cp_inst.add_file(self.file)
        cp_inst["R"] = self.R

        start_time = time.time()
        result_all = cp_inst.solve(optimisation_level=2, all_solutions=True)

        end_time = time.time()
        print("FINISHED")
        print("Total time taken: %.2f seconds" % (end_time - start_time))

        solutions = []
        for result in result_all:
            x0 = []
            for i in range(64):
                for j in range(1, 32):
                    if result.x[0][i][j] == True:
                        x0.append([i, j])
            x = [[] for _ in range(64)]
            for i in range(64):
                for j in range(32):
                    if result.x[self.R][i][j] == True:
                        x[i].append(j)
            solutions.append((x0, x))
        return solutions


if __name__ == "__main__":
    init = [
        [[[1, 3.0]], [[0, 11], [7, 1], [10, 2], [17, 2], [39, 8], [41, 1], [61, 8]]],
        [[[17, 2.0]], [[0, 17], [7, 1], [19, 16], [28, 16], [41, 1]]],
        [[[12, 2.0]], [[0, 16], [19, 16], [28, 16]]],
        [[[4, 2.0]], [[0, 6], [1, 4], [6, 4], [10, 2], [17, 2]]],
        [[[16, 2.0]], [[0, 24], [19, 16], [28, 16], [39, 8], [61, 8]]],
        [[[6, 4.0]], [[0, 1], [7, 1], [41, 1]]],
        [[[7, 3.0]], [[0, 2], [10, 2], [17, 2]]],
        [[[19, 2.0]], [[0, 4], [1, 4], [6, 4]]],
        [[[15, 3.0]], [[0, 8], [39, 8], [61, 8]]],
        [[[27, 4.0]], [[0, 5], [1, 4], [6, 4], [7, 1], [41, 1]]],
    ]

    R = 2

    model = AsconMiddleLayer(R)
    for i in init:
        solutions = model.solve(i[1])
        result_dir = os.path.join(BASE_DIR, "..", "DL_PSP", "Middle_layer", "result")
        os.makedirs(result_dir, exist_ok=True)
        out_file = os.path.join(result_dir, f"middle_layer_{R}_{i[0][0][0]}.txt")

        with open(out_file, "w", encoding="utf-8") as file:
            file.write("SATISFIED\n")
            file.write("Time taken: ---\n")
            file.write("R: %d\n" % R)
            for idx, (x0, x) in enumerate(solutions):
                file.write("\n\n--------------------------------------------------------------\n")
                file.write("Solution %d\n" % idx)
                file.write("x0 = %s\n" % x0)
                for i in range(64):
                    file.write("x:%d \t%s\n" % (i, x[i]))
                    file.write("--------------------------------------------------------------\n")
