import math


class Relation:
    permutation = [31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10, 15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26]
    sbox = [12, 10, 13, 3, 14, 11, 15, 7, 8, 9, 1, 5, 0, 2, 4, 6]

    ddt = [[0 for _ in range(16)] for _ in range(16)]
    for _x in range(16):
        for _y in range(16):
            ddt[_x ^ _y][sbox[_x] ^ sbox[_y]] += 1
    for _i in range(16):
        for _j in range(16):
            ddt[_i][_j] = float(ddt[_i][_j]) / 16.0

    permutation_rev = [0 for _ in range(32)]
    for _i in range(32):
        permutation_rev[permutation[_i]] = _i

    def __init__(self, trail, input):
        self.trail = trail
        self.input_cell = input[0]
        self.input_val = input[1]
        self.x = {}

    def sbox_operation(self, cell):
        temp = [0.0 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                temp[j] += cell[i] * self.ddt[i][j]
        return temp

    def xor_operation(self, cell_a, cell_b):
        temp = [0.0 for _ in range(16)]
        for x0 in range(16):
            for x1 in range(16):
                temp[x0 ^ x1] += cell_a[x0] * cell_b[x1]
        return temp

    def propagate(self):
        trail = self.trail
        for r in range(len(trail)):
            for a in trail[r]:
                self.x[f"{r}, {a}"] = [0.0 for _ in range(16)]
                self.x[f"{r}, {a}"][0] = 1.0
        self.x[f"0, {self.input_cell}"][self.input_val] = 1.0
        self.x[f"0, {self.input_cell}"][0] = 0.0
        for r in range(len(trail) - 1):
            for a in trail[r + 1]:
                idx = self.permutation_rev[a]
                if idx % 2 == 0:
                    self.x[f"{r + 1}, {a}"] = self.x[f"{r}, {idx}"]
                else:
                    key_left = f"{r}, {idx - 1}"
                    key_right = f"{r}, {idx}"
                    if key_left in self.x and key_right in self.x:
                        self.x[f"{r + 1}, {a}"] = self.xor_operation(self.sbox_operation(self.x[key_left]), self.x[key_right])
                    elif key_left in self.x:
                        self.x[f"{r + 1}, {a}"] = self.sbox_operation(self.x[key_left])
                    elif key_right in self.x:
                        self.x[f"{r + 1}, {a}"] = self.x[key_right]
                    else:
                        print(f"Error: No input for cell {a} at round {r + 1}")

    def analyze(self):
        self.propagate()
        trail = self.trail
        result = set()
        cell = trail[-1][0]
        output = [[0, 0] for _ in range(16)]
        for i in range(16):
            i_bit = [int(x) for x in f"{i:04b}"]
            for j in range(16):
                j_bit = [int(x) for x in f"{j:04b}"]
                hd = sum([i_bit[k] * j_bit[k] for k in range(4)]) & 1
                output[i][hd] += self.x[f"{len(trail) - 1}, {cell}"][j]
        correlations = []
        for i in range(1, 16):
            if output[i][0] != output[i][1]:
                cor = math.log2(float(abs(output[i][0] - output[i][1]) / (output[i][0] + output[i][1])))
                correlations.append((i, cor))
        if correlations:
            max_cor = max(cor for _, cor in correlations)
            for i, cor in correlations:
                if cor == max_cor:
                    result.add((cell, i, cor))
        return result


if __name__ == "__main__":
    trail = [[11], [0], [6, 31], [8, 21, 26], [2, 16, 19, 28], [9, 14, 15, 20, 29, 30], [2, 7, 10, 20, 28], [3, 8, 14, 20, 28], [9, 10, 14, 17, 27], [2, 3, 16, 22, 23], [14, 15, 24], [10, 11], [0]]
    input = [11, 15]
    relation = Relation(trail, input)
    result = relation.analyze()
    for cell, i, cor in sorted(result):
        print(f"input:[{relation.input_cell}, {relation.input_val}] -> output:[{cell}, {i}]  r: 2^{cor:.8f}")