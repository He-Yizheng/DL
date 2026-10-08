import math
import time
import numpy as np

start_time = time.time()

SIZE = 256

# SKINNY-128 Sbox (8-bit)
sbox = np.array([
    0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
    0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
    0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
    0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
    0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
    0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
    0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
    0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
    0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
    0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
    0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
    0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
    0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
    0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
    0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
    0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff,
], dtype=np.uint8)

permutation = [
    0, 1, 2, 3,
    7, 4, 5, 6,
    10, 11, 8, 9,
    13, 14, 15, 12,
]

idx = np.arange(SIZE)
rows = (idx[:, None] ^ idx[None, :]).ravel()
cols = (sbox[:, None] ^ sbox[None, :]).ravel()
ddt = np.zeros((SIZE, SIZE))
np.add.at(ddt, (rows, cols), 1.0)
ddt /= SIZE


def sbox_layer(cell):
    # temp[j] = sum_i cell[i] * ddt[i][j]
    return ddt.T @ cell


def fwt(a):
    a = np.asarray(a, dtype=np.float64)
    n = 1
    while n < SIZE:
        a = a.reshape(-1, 2 * n)
        x, y = a[:, :n], a[:, n:]
        a = np.hstack((x + y, x - y)).reshape(-1)
        n *= 2
    return a


def xor_layer(cell0, cell1):
    # temp[i ^ j] += cell0[i] * cell1[j]
    return fwt(fwt(cell0) * fwt(cell1)) / SIZE


def zero_distribution():
    cell = np.zeros(SIZE, dtype=np.float64)
    cell[0] = 1.0
    return cell


def get_cell(state, round_index, cell_index):
    key = (round_index, cell_index)
    if key not in state:
        state[key] = zero_distribution()
    return state[key]


def propagate_sparse_round(state, round_index, next_active_cells):
    current_cells = [
        cell_index
        for stored_round, cell_index in state
        if stored_round == round_index
    ]
    for cell_index in current_cells:
        state[(round_index, cell_index)] = sbox_layer(
            state[(round_index, cell_index)]
        )

    for i in range(4):
        next_cell = i
        if next_cell in next_active_cells:
            a = get_cell(state, round_index, permutation[i])
            b = get_cell(state, round_index, permutation[i + 8])
            c = get_cell(state, round_index, permutation[i + 12])
            state[(round_index + 1, next_cell)] = xor_layer(
                xor_layer(a, b), c
            )

        next_cell = i + 4
        if next_cell in next_active_cells:
            state[(round_index + 1, next_cell)] = get_cell(
                state, round_index, permutation[i]
            ).copy()

        next_cell = i + 8
        if next_cell in next_active_cells:
            a = get_cell(state, round_index, permutation[i + 4])
            b = get_cell(state, round_index, permutation[i + 8])
            state[(round_index + 1, next_cell)] = xor_layer(a, b)

        next_cell = i + 12
        if next_cell in next_active_cells:
            a = get_cell(state, round_index, permutation[i])
            b = get_cell(state, round_index, permutation[i + 8])
            state[(round_index + 1, next_cell)] = xor_layer(a, b)


# trail = [[15], [2], [2, 6, 14], [1, 2, 11], [2, 5, 9], [2, 3, 10, 15], [2, 8, 15], [2]]
trail = [[12], [3], [3, 7], [7, 8, 10, 14], [0, 1, 8, 10, 11], [0, 10, 13], [0], [4]]

INPUT_CELL = 12
INPUT_DIFF = 0x02
OUTPUT_CELL = 4


ij = idx[:, None] & idx[None, :]
parity = np.zeros((SIZE, SIZE), dtype=np.int64)
for bit_index in range(8):
    parity += (ij >> bit_index) & 1
parity &= 1
PAR0 = (parity == 0).astype(np.float64)
PAR1 = (parity == 1).astype(np.float64)


if __name__ == "__main__":
    state = {}
    for round_index, active_cells in enumerate(trail):
        for cell_index in active_cells:
            state[(round_index, cell_index)] = zero_distribution()

    state[(0, INPUT_CELL)][0] = 0.0
    state[(0, INPUT_CELL)][INPUT_DIFF] = 1.0

    for round_index in range(len(trail) - 1):
        propagate_sparse_round(state, round_index, set(trail[round_index + 1]))
        print(
            f"round {round_index}: active output cells "
            f"{trail[round_index + 1]}"
        )

    output_distribution = state[(len(trail) - 1, OUTPUT_CELL)]
    output_distribution[np.abs(output_distribution) < 1e-15] = 0.0

    print("\nFinal output distribution:")
    print(output_distribution.tolist())
    print("\nOutput-mask correlations:")

    out0 = PAR0 @ output_distribution
    out1 = PAR1 @ output_distribution
    for mask in range(1, SIZE):
        denominator = out0[mask] + out1[mask]
        correlation = 0.0
        if denominator != 0.0:
            correlation = abs(out0[mask] - out1[mask]) / denominator

        print(
            f"{mask:08b}:\t{out0[mask]:.8f}\t{out1[mask]:.8f}\t",
            end="",
        )
        if correlation > 0.0:
            print(f"r: 2^{math.log2(correlation):.8f}")
        else:
            print("r: 2^-infinite")

    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
