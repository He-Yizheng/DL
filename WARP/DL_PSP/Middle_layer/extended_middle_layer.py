import re
import os
import time
import minizinc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LINE_RE = re.compile(r"^(\d+):\s*input\[(\d+),\s*(\d+)\]\s+cell:\s*(\d+)\s*$")

MODEL = r'''
int: R;
int: INPUT_CELL;
int: DEFECT_CELL;

array[0..R, 0..31] of var bool: x;
array[0..R, 0..31] of var bool: y;

array[0..31] of int: permutation = array1d(0..31, [
    31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10,
    15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26
]);

constraint forall(r in 0..(R - 1), l in 0..15)(
    x[r + 1, permutation[l * 2]] = x[r, l * 2]
    /\
    x[r + 1, permutation[l * 2 + 1]] = (x[r, l * 2] \/ x[r, l * 2 + 1])
);

constraint forall(r in 2..(R - 1), l in 0..15)(
    y[r + 1, l * 2] = (y[r, permutation[l * 2]] \/ y[r, permutation[l * 2 + 1]])
    /\
    y[r + 1, l * 2 + 1] = y[r, permutation[l * 2 + 1]]
);

constraint forall(r in 0..1, l in 0..15)(
    y[r, permutation[l * 2]] = y[r + 1, l * 2]
    /\
    y[r, permutation[l * 2 + 1]] = (y[r + 1, l * 2] \/ y[r + 1, l * 2 + 1])
);

constraint forall(l in 0..31)(
    if l == INPUT_CELL then x[0, l] = true
    else x[0, l] = false
    endif
);

constraint forall(l in 0..31)(
    if l == DEFECT_CELL then y[2, l] = true
    else y[2, l] = false
    endif
);

solve satisfy;
'''

MODEL_LAST = r'''
int: R;
int: INPUT_CELL;
int: OUTPUT_COUNT;
array[0..OUTPUT_COUNT-1] of int: OUTPUT_CELLS;

array[0..R, 0..31] of var bool: x;
array[0..R, 0..31] of var bool: y;

array[0..31] of int: permutation = array1d(0..31, [
    31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10,
    15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26
]);

constraint forall(r in 0..(R - 1), l in 0..15)(
    x[r + 1, permutation[l * 2]] = x[r, l * 2]
    /\
    x[r + 1, permutation[l * 2 + 1]] = (x[r, l * 2] \/ x[r, l * 2 + 1])
);

constraint forall(r in 0..(R - 1), l in 0..15)(
    y[r + 1, l * 2] = (y[r, permutation[l * 2]] \/ y[r, permutation[l * 2 + 1]])
    /\
    y[r + 1, l * 2 + 1] = y[r, permutation[l * 2 + 1]]
);

constraint forall(l in 0..31)(
    if l == INPUT_CELL then x[0, l] = true
    else x[0, l] = false
    endif
);

constraint forall(l in 0..31)(
    if exists(i in 0..OUTPUT_COUNT-1)(l == OUTPUT_CELLS[i]) then y[0, l] = true
    else y[0, l] = false
    endif
);

solve satisfy;
'''


class ExtendedMiddleLayer:
    def __init__(self, defects=None, defect_file=None, R=10):
        self.defects = defects
        self.defect_file = defect_file
        self.R = R

    def run(self):
        if self.defects is None:
            if self.defect_file is None:
                defect_file = os.path.join(BASE_DIR, "result", "defect_cells.txt")
            else:
                defect_file = self.defect_file
            defects = []
            with open(defect_file, "r", encoding="utf-8") as file:
                for line in file:
                    match = LINE_RE.match(line.strip())
                    if match:
                        defects.append((int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))))
        else:
            defects = self.defects

        cp_solver = minizinc.Solver.lookup("com.google.ortools.sat")
        cp_model = minizinc.Model()
        cp_model.add_string(MODEL)
        cp_last_model = minizinc.Model()
        cp_last_model.add_string(MODEL_LAST)

        start_time = time.time()
        trails = []
        idx = 1
        for no, input_cell, input_val, defect_cell in defects:
            cp_inst = minizinc.Instance(solver=cp_solver, model=cp_model)
            cp_inst["R"] = self.R
            cp_inst["INPUT_CELL"] = input_cell
            cp_inst["DEFECT_CELL"] = defect_cell
            result = cp_inst.solve(optimisation_level=2)

            x = result["x"]
            y = result["y"]
            trail = self.build_trail(x, y, self.R)
            last_cells = trail[-1]

            for last_cell in last_cells:
                cp_last_inst = minizinc.Instance(solver=cp_solver, model=cp_last_model)
                cp_last_inst["R"] = self.R
                cp_last_inst["INPUT_CELL"] = input_cell
                cp_last_inst["OUTPUT_COUNT"] = 1
                cp_last_inst["OUTPUT_CELLS"] = [last_cell]
                result_last = cp_last_inst.solve(optimisation_level=2)

                x_last = result_last["x"]
                y_last = result_last["y"]
                trail_last = self.build_trail(x_last, y_last, self.R)
                trails.append((idx, input_cell, input_val, trail_last))
                idx += 1

        end_time = time.time()
        print("FINISHED")
        print("Total time taken: %.2f seconds" % (end_time - start_time))
        return trails

    @staticmethod
    def build_trail(x, y, R):
        return [[i for i in range(32) if x[r][i] and y[R - r][i]] for r in range(R + 1)]


if __name__ == "__main__":
    out_file = os.path.join(BASE_DIR, "result", "extended_middle_layer.txt")
    trails = ExtendedMiddleLayer(defect_file=None, R=12).run()
    with open(out_file, "w", encoding="utf-8") as file:
        for idx, input_cell, input_val, trail in trails:
            file.write("%d: input[%d, %d]: %s\n" % (idx, input_cell, input_val, trail))
