import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DL.WARP.Middle_layer_CP.warp import WarpMiddleLayer
from DL.WARP.DL_BSP.distinguisher import BSPDistinguisher
from DL.WARP.DL_PSP.distinguisher import PSPDistinguisher


class WARP:
    def run(self):
        print("1: Generate middle layer BSP result")
        print("2: DL-BSP")
        print("3: DL-PSP")
        choice = input("Choose (1/2/3): ").strip()

        if choice == "1":
            R = int(input("R: "))
            target = input("1: BSP, 2: PSP: ").strip()
            if target == "1":
                result_dir = os.path.join(BASE_DIR, "DL_BSP", "Middle_layer", "result")
            else:
                result_dir = os.path.join(BASE_DIR, "DL_PSP", "Middle_layer", "result")
            out_file = os.path.join(result_dir, "middle_layer_%d.txt" % R)

            start_time = time.time()
            solutions = self.generate_middle_layer(R)
            end_time = time.time()
            with open(out_file, "w", encoding="utf-8") as file:
                file.write("SATISFIED\n")
                file.write("Time taken: %.2f seconds\n" % (end_time - start_time))
                file.write("R: %d\n" % R)
                for idx, (x0, x) in enumerate(solutions):
                    file.write("\n\n--------------------------------------------------------------\n")
                    file.write("Solution %d\n" % idx)
                    file.write("x0 = %s\n" % x0)
                    for i in range(32):
                        file.write("x:%d \t%s\n" % (i, x[i]))
                        file.write("--------------------------------------------------------------\n")
            print("Written to: %s" % out_file)

        elif choice == "2":
            diff_rounds = int(input("diff_rounds: "))
            middle_rounds = int(input("middle_rounds: "))
            lin_rounds = int(input("lin_rounds: "))
            self.run_bsp(diff_rounds=diff_rounds, middle_rounds=middle_rounds, lin_rounds=lin_rounds)
            print("Done.")

        elif choice == "3":
            diff_rounds = int(input("diff_rounds: "))
            middle_rounds = int(input("middle_rounds: "))
            extended_rounds = int(input("extended_rounds: "))
            lin_rounds = int(input("lin_rounds: "))
            self.run_psp(diff_rounds=diff_rounds, middle_rounds=middle_rounds, extended_rounds=extended_rounds, lin_rounds=lin_rounds)
            print("Done.")
        else:
            print("Invalid choice")
            return 1
        return 0

    def generate_middle_layer(self, R):
        return WarpMiddleLayer(R).solve()

    def run_bsp(self, diff_rounds=6, middle_rounds=11, lin_rounds=6):
        BSPDistinguisher(diff_rounds=diff_rounds, middle_rounds=middle_rounds, lin_rounds=lin_rounds).build()

    def run_psp(self, diff_rounds=6, middle_rounds=10, extended_rounds=2, lin_rounds=6):
        PSPDistinguisher(diff_rounds=diff_rounds, middle_rounds=middle_rounds, extended_rounds=extended_rounds, lin_rounds=lin_rounds).build()


if __name__ == "__main__":
    sys.exit(WARP().run())
