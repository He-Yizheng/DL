import math
import time


CELL_BITS = 5
CELL_SIZE = 1 << CELL_BITS
STATE_SIZE = 64

SBOX = [
    4, 11, 31, 20, 26, 21, 9, 2,
    27, 5, 8, 18, 29, 3, 6, 28,
    30, 19, 7, 14, 0, 13, 17, 24,
    16, 12, 1, 25, 22, 10, 15, 23,
]

ROTATION_OFFSETS = [
    (45, 36),
    (3, 25),
    (63, 58),
    (54, 47),
    (57, 23),
]


def build_ddt():
    ddt = [
        [0.0 for _ in range(CELL_SIZE)]
        for _ in range(CELL_SIZE)
    ]

    for x0 in range(CELL_SIZE):
        for x1 in range(CELL_SIZE):
            input_difference = x0 ^ x1
            output_difference = SBOX[x0] ^ SBOX[x1]
            ddt[input_difference][output_difference] += 1.0

    for input_difference in range(CELL_SIZE):
        for output_difference in range(CELL_SIZE):
            ddt[input_difference][output_difference] /= CELL_SIZE

    return ddt


DDT = build_ddt()


def zero_cell_distribution():
    distribution = [0.0 for _ in range(CELL_SIZE)]
    distribution[0] = 1.0
    return distribution


def sbox_layer_cell(cell):
    output = [0.0 for _ in range(CELL_SIZE)]

    for input_difference in range(CELL_SIZE):
        for output_difference in range(CELL_SIZE):
            output[output_difference] += (
                cell[input_difference]
                * DDT[input_difference][output_difference]
            )

    return output


def sbox_layer_bits(cell):
    cell_output = sbox_layer_cell(cell)

    bit_output = [
        [0.0, 0.0]
        for _ in range(CELL_BITS)
    ]

    for difference in range(CELL_SIZE):
        for bit_index in range(CELL_BITS):
            bit_value = (
                difference
                >> (CELL_BITS - 1 - bit_index)
            ) & 1

            bit_output[bit_index][bit_value] += cell_output[difference]

    return bit_output


def xor_bit_distributions(distribution0, distribution1):
    output = [0.0, 0.0]

    for value0 in range(2):
        for value1 in range(2):
            output[value0 ^ value1] += (
                distribution0[value0]
                * distribution1[value1]
            )

    return output


def combine_optional_sources(
    state,
    source0_key,
    source1_key,
    bit_index,
):
    source0_exists = source0_key in state
    source1_exists = source1_key in state

    if source0_exists and source1_exists:
        return xor_bit_distributions(
            state[source0_key][bit_index],
            state[source1_key][bit_index],
        )

    if source0_exists:
        return state[source0_key][bit_index]

    if source1_exists:
        return state[source1_key][bit_index]

    return [1.0, 0.0]


def bit_to_cell(direct_cell, rotated_bits):
    output = [0.0 for _ in range(CELL_SIZE)]

    for direct_difference in range(CELL_SIZE):
        for rotated_difference in range(CELL_SIZE):
            rotated_probability = 1.0

            for bit_index in range(CELL_BITS):
                bit_value = (
                    rotated_difference
                    >> (CELL_BITS - 1 - bit_index)
                ) & 1

                rotated_probability *= rotated_bits[bit_index][bit_value]

            output[direct_difference ^ rotated_difference] += (
                direct_cell[direct_difference]
                * rotated_probability
            )

    return output


def cell_key(round_index, cell_index):
    return f"{round_index}_{cell_index}"


def sbox_bit_key(round_index, cell_index):
    return f"{round_index}_{cell_index}_sb"


def sbox_cell_key(round_index, cell_index):
    return f"{round_index}_{cell_index}_sb_cell"


def initialize_state(trail, input_differences):
    state = {}

    for round_index, active_cells in enumerate(trail):
        for cell_index in active_cells:
            state[cell_key(round_index, cell_index)] = [
                0.0 for _ in range(CELL_SIZE)
            ]

    for cell_index, difference in input_differences.items():
        state[cell_key(0, cell_index)][difference] = 1.0

    return state


def propagate_sbox_layer(state, round_index, trail):
    for cell_index in trail[round_index]:
        input_distribution = state[cell_key(round_index, cell_index)]

        state[sbox_bit_key(round_index, cell_index)] = (
            sbox_layer_bits(input_distribution)
        )

        state[sbox_cell_key(round_index, cell_index)] = (
            sbox_layer_cell(input_distribution)
        )


def propagate_linear_layer(state, round_index, trail):
    for output_cell in trail[round_index + 1]:
        rotated_bits = []

        for bit_index, offsets in enumerate(ROTATION_OFFSETS):
            offset0, offset1 = offsets

            source0 = (output_cell + offset0) % STATE_SIZE
            source1 = (output_cell + offset1) % STATE_SIZE

            rotated_bits.append(
                combine_optional_sources(
                    state,
                    sbox_bit_key(round_index, source0),
                    sbox_bit_key(round_index, source1),
                    bit_index,
                )
            )

        direct_key = sbox_cell_key(round_index, output_cell)

        if direct_key in state:
            direct_cell = state[direct_key]
        else:
            direct_cell = zero_cell_distribution()

        state[cell_key(round_index + 1, output_cell)] = bit_to_cell(
            direct_cell,
            rotated_bits,
        )


def print_round_state(state, round_index):
    sbox_cells = {
        key: value
        for key, value in state.items()
        if key.startswith(f"{round_index}_")
        and key.endswith("_sb_cell")
    }

    sbox_bits = {
        key: value
        for key, value in state.items()
        if key.startswith(f"{round_index}_")
        and key.endswith("_sb")
    }

    next_cells = {
        key: value
        for key, value in state.items()
        if key.startswith(f"{round_index + 1}_")
        and not key.endswith("_sb")
        and not key.endswith("_sb_cell")
    }

    print(sbox_cells)
    print(sbox_bits)
    print(next_cells)


def compute_output_correlations(output_distribution):
    output = [
        [0.0, 0.0]
        for _ in range(CELL_SIZE)
    ]

    for mask in range(CELL_SIZE):
        mask_bits = [
            int(bit)
            for bit in f"{mask:05b}"
        ]

        for difference in range(CELL_SIZE):
            difference_bits = [
                int(bit)
                for bit in f"{difference:05b}"
            ]

            parity = sum(
                mask_bits[bit_index]
                * difference_bits[bit_index]
                for bit_index in range(CELL_BITS)
            ) & 1

            output[mask][parity] += output_distribution[difference]

    return output


def print_output_correlations(output):
    for mask in range(1, CELL_SIZE):
        probability0 = output[mask][0]
        probability1 = output[mask][1]

        print(
            f"{mask:05b}:\t"
            f"{probability0:.8f}\t"
            f"{probability1:.8f}\t",
            end="",
        )

        if probability0 != probability1:
            correlation = abs(
                probability0 - probability1
            ) / (
                probability0 + probability1
            )

            print(
                f"r: 2^{math.log2(float(correlation)):.8f}"
            )
        else:
            print("r: 2^-infinite")


def main(trail, input_differences):
    start_time = time.time()

    state = initialize_state(
        trail,
        input_differences,
    )

    for round_index in range(len(trail) - 1):
        propagate_sbox_layer(
            state,
            round_index,
            trail,
        )

        propagate_linear_layer(
            state,
            round_index,
            trail,
        )

        print_round_state(
            state,
            round_index,
        )

    output_key = cell_key(
        len(trail) - 1,
        trail[-1][0],
    )
    output_distribution = state[output_key]

    print("\nFinal output:")
    print(output_distribution)
    print()

    correlations = compute_output_correlations(
        output_distribution
    )
    print_output_correlations(correlations)

    print()
    print(
        "Time taken: %.2f seconds"
        % (time.time() - start_time)
    )


if __name__ == "__main__":
    trail = [[32, 51, 60], [2, 3, 4, 7, 9, 13, 15, 24, 26, 29, 33, 35, 42, 49, 51, 52, 57], [8, 10, 17, 20, 21, 26, 27, 30, 50, 52, 63], [27]]
    
    input_differences = {
        32: 0x02,
        51: 0x02,
        60: 0x02,
    }
    main(trail, input_differences)
