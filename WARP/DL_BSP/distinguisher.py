import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DL.WARP.DL_BSP.MILP.diff import Diff
from DL.WARP.DL_BSP.MILP.lin import Lin
from DL.WARP.DL_BSP.Middle_layer.defected_point import DefectPointParser


class BSPDistinguisher:
    def __init__(self, middle_file=None, out_file=None, diff_rounds=6, middle_rounds=11, lin_rounds=6):
        self.diff_rounds = diff_rounds
        self.middle_rounds = middle_rounds
        self.lin_rounds = lin_rounds
        self.middle_file = middle_file if middle_file is not None else os.path.join(
            BASE_DIR, "Middle_layer", "result", "middle_layer_%d.txt" % middle_rounds)
        self.out_file = out_file if out_file is not None else os.path.join(BASE_DIR, "distinguishers.txt")

    def build(self):
        os.chdir(PROJECT_ROOT)

        entries, _, _ = DefectPointParser(self.middle_file).parse()

        count = 0
        with open(self.out_file, "w", encoding="utf-8") as file:
            for idx, (_, input_cell, input_val, cell, point) in enumerate(entries, start=1):
                d = Diff(self.diff_rounds, input_cell, input_val)
                d.make()
                diff_obj, delta_i = d.solve()

                mask = self.mask_of_bit(point)
                l = Lin(self.lin_rounds, cell, mask)
                l.make()
                lin_obj, lambda_o = l.solve()

                if delta_i is None or lambda_o is None or diff_obj is None or lin_obj is None:
                    continue

                delta_m = self.state_to_hex(input_cell, input_val)
                lambda_m = self.state_to_hex(cell, mask)

                file.write("distinguisher %d:\n" % idx)
                file.write("i: %s\n" % delta_i)
                file.write("m: %s\n" % delta_m)
                file.write("m: %s\n" % lambda_m)
                file.write("o: %s\n" % lambda_o)
                file.write("p=2^-%d, r=1, q=2^-%d, cor=2^-%d\n" % (diff_obj, lin_obj, diff_obj + 2 * lin_obj))
                file.write("\n")
                count += 1

        print("distinguishers: %d" % count)

    @staticmethod
    def mask_of_bit(bit, bit_width=4):
        return 1 << (bit_width - bit)

    @staticmethod
    def state_to_hex(cell_index, cell_value, num_cells=32):
        state = [0] * num_cells
        state[cell_index] = cell_value
        return "".join("%x" % v for v in state)


if __name__ == "__main__":
    BSPDistinguisher().build()
