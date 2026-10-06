import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIL_RE = re.compile(r"^(\d+): input\[(\d+), (\d+)\]: (\[.*\])$")

from DL.WARP.DL_PSP.MILP.diff import Diff
from DL.WARP.DL_PSP.MILP.lin import Lin
from DL.WARP.DL_PSP.Middle_layer.defect_cell import DefectCellParser
from DL.WARP.DL_PSP.Middle_layer.extended_middle_layer import ExtendedMiddleLayer
from DL.WARP.DL_PSP.Middle_layer.relation import Relation


class PSPDistinguisher:
    def __init__(self, middle_file=None, out_file=None, diff_rounds=6, middle_rounds=10, extended_rounds=2, lin_rounds=6):
        self.diff_rounds = diff_rounds
        self.middle_rounds = middle_rounds
        self.extended_rounds = extended_rounds
        self.lin_rounds = lin_rounds
        self.middle_file = middle_file if middle_file is not None else os.path.join(
            BASE_DIR, "Middle_layer", "result", "middle_layer_%d.txt" % middle_rounds)
        self.out_file = out_file if out_file is not None else os.path.join(BASE_DIR, "distinguishers.txt")

    def build(self):
        os.chdir(PROJECT_ROOT)

        cells = DefectCellParser(self.middle_file).parse()
        trails = ExtendedMiddleLayer(defects=cells, R=self.middle_rounds + self.extended_rounds).run()

        count = 0
        dist_no = 1
        with open(self.out_file, "w", encoding="utf-8") as file:
            for no, input_cell, input_val, trail in trails:
                relation = Relation(trail, [input_cell, input_val])
                outputs = relation.analyze()

                for out_cell, out_val, cor in sorted(outputs):
                    d = Diff(self.diff_rounds, input_cell, input_val)
                    d.make()
                    diff_obj, delta_i = d.solve()

                    l = Lin(self.lin_rounds, out_cell, out_val)
                    l.make()
                    lin_obj, lambda_o = l.solve()

                    if delta_i is None or lambda_o is None or diff_obj is None or lin_obj is None:
                        continue

                    delta_m = self.state_to_hex(input_cell, input_val)
                    lambda_m = self.state_to_hex(out_cell, out_val)

                    file.write("distinguisher %d:\n" % dist_no)
                    dist_no += 1
                    file.write("i: %s\n" % delta_i)
                    file.write("m: %s\n" % delta_m)
                    file.write("m: %s\n" % lambda_m)
                    file.write("o: %s\n" % lambda_o)
                    r_val = round(-cor, 2)
                    cor_val = round(diff_obj + r_val + 2 * lin_obj, 2)
                    file.write("p=2^-%.2f, r=2^-%.2f, q=2^-%.2f, cor=2^-%.2f\n" % (diff_obj, r_val, lin_obj, cor_val))
                    file.write("\n")
                    print("distinguisher %d: cor=2^-%.2f" % (dist_no - 1, cor_val))
                    count += 1

        print("distinguishers: %d" % count)

    @staticmethod
    def parse_trails(trail_file):
        trails = []
        with open(trail_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                match = TRAIL_RE.match(line)
                if not match:
                    continue
                no = int(match.group(1))
                input_cell = int(match.group(2))
                input_val = int(match.group(3))
                body = match.group(4)
                trail = []
                for inner in re.findall(r"\[[^\]]*\]", body):
                    vals = [int(v.strip()) for v in inner.strip("[]").split(",") if v.strip()]
                    trail.append(vals)
                trails.append((no, input_cell, input_val, trail))
        return trails

    @staticmethod
    def state_to_hex(cell_index, cell_value, num_cells=32):
        state = [0] * num_cells
        state[cell_index] = cell_value
        return "".join("%x" % v for v in state)


if __name__ == "__main__":
    PSPDistinguisher().build()
