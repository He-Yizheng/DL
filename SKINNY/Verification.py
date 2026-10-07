import math
import time

import numpy as np
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)


STATE_CELLS = 16
GPU_DEVICE = 0


@cuda.jit(device=True)
def sbox_function(x, x_s, sbox):
    for i in range(STATE_CELLS):
        x_s[i] = sbox[x[i]]


@cuda.jit(device=True)
def linear_function(x_s, x, permutation):
    for i in range(4):
        x[i] = (
            x_s[permutation[i]]
            ^ x_s[permutation[i + 8]]
            ^ x_s[permutation[i + 12]]
        )
        x[i + 4] = x_s[permutation[i]]
        x[i + 8] = x_s[permutation[i + 4]] ^ x_s[permutation[i + 8]]
        x[i + 12] = x_s[permutation[i]] ^ x_s[permutation[i + 8]]


@cuda.jit(device=True)
def round_function(x, x_s, sbox, permutation):
    sbox_function(x, x_s, sbox)
    linear_function(x_s, x, permutation)


@cuda.jit(device=True)
def parity8(value):
    value ^= value >> 4
    value ^= value >> 2
    value ^= value >> 1
    return value & 1


@cuda.jit(device=True)
def masked_state_parity(x, mask):
    value = 0
    for i in range(STATE_CELLS):
        value ^= parity8(x[i] & mask[i])
    return value


@cuda.jit
def simulation_kernel(
    rng_states,
    iterations_per_thread,
    sbox,
    permutation,
    r_d,
    r_m,
    r_l,
    diff_trail,
    lin_trail,
    count,
    linear_round_count,
):

    thread_id = cuda.grid(1)

    input0 = cuda.local.array(STATE_CELLS, dtype=np.uint8)
    input1 = cuda.local.array(STATE_CELLS, dtype=np.uint8)
    input0_s = cuda.local.array(STATE_CELLS, dtype=np.uint8)
    input1_s = cuda.local.array(STATE_CELLS, dtype=np.uint8)

    for _ in range(iterations_per_thread):
        for i in range(STATE_CELLS):
            input0[i] = int(
                xoroshiro128p_uniform_float32(rng_states, thread_id) * 256
            )
            input1[i] = input0[i] ^ diff_trail[0, i]

        follows_diff = True
        for r in range(r_d):
            round_function(input0, input0_s, sbox, permutation)
            round_function(input1, input1_s, sbox, permutation)

            for i in range(STATE_CELLS):
                if (input0[i] ^ input1[i]) != diff_trail[r + 1, i]:
                    follows_diff = False
                    break

            if not follows_diff:
                break

        if not follows_diff:
            continue

        cuda.atomic.add(count, 0, 1)

        for _ in range(r_m):
            round_function(input0, input0_s, sbox, permutation)
            round_function(input1, input1_s, sbox, permutation)

        parity0 = masked_state_parity(input0, lin_trail[0])
        parity1 = masked_state_parity(input1, lin_trail[0])
        if (parity0 ^ parity1) == 1:
            cuda.atomic.add(count, 1, 1)

        for r in range(r_l):
            round_parity0 = masked_state_parity(input0, lin_trail[r])
            round_parity1 = masked_state_parity(input1, lin_trail[r])

            round_function(input0, input0_s, sbox, permutation)
            round_function(input1, input1_s, sbox, permutation)

            round_parity0 ^= masked_state_parity(input0, lin_trail[r + 1])
            round_parity1 ^= masked_state_parity(input1, lin_trail[r + 1])
            if round_parity0 == 1:
                cuda.atomic.add(linear_round_count, (0, r), 1)
            if round_parity1 == 1:
                cuda.atomic.add(linear_round_count, (1, r), 1)

        parity0 ^= masked_state_parity(input0, lin_trail[r_l])
        parity1 ^= masked_state_parity(input1, lin_trail[r_l])
        if parity0 == 1:
            cuda.atomic.add(count, 2, 1)
        if parity1 == 1:
            cuda.atomic.add(count, 3, 1)


def correlation(one_count, total_count):
    if total_count == 0:
        return 0.0
    return abs(total_count - 2 * one_count) / total_count


def log2_value(value):
    if value == 0.0:
        return float("-inf")
    return math.log2(value)


if __name__ == "__main__":
    cuda.select_device(GPU_DEVICE)

    sbox = np.array(
        [
            0x65, 0x4C, 0x6A, 0x42, 0x4B, 0x63, 0x43, 0x6B,
            0x55, 0x75, 0x5A, 0x7A, 0x53, 0x73, 0x5B, 0x7B,
            0x35, 0x8C, 0x3A, 0x81, 0x89, 0x33, 0x80, 0x3B,
            0x95, 0x25, 0x98, 0x2A, 0x90, 0x23, 0x99, 0x2B,
            0xE5, 0xCC, 0xE8, 0xC1, 0xC9, 0xE0, 0xC0, 0xE9,
            0xD5, 0xF5, 0xD8, 0xF8, 0xD0, 0xF0, 0xD9, 0xF9,
            0xA5, 0x1C, 0xA8, 0x12, 0x1B, 0xA0, 0x13, 0xA9,
            0x05, 0xB5, 0x0A, 0xB8, 0x03, 0xB0, 0x0B, 0xB9,
            0x32, 0x88, 0x3C, 0x85, 0x8D, 0x34, 0x84, 0x3D,
            0x91, 0x22, 0x9C, 0x2C, 0x94, 0x24, 0x9D, 0x2D,
            0x62, 0x4A, 0x6C, 0x45, 0x4D, 0x64, 0x44, 0x6D,
            0x52, 0x72, 0x5C, 0x7C, 0x54, 0x74, 0x5D, 0x7D,
            0xA1, 0x1A, 0xAC, 0x15, 0x1D, 0xA4, 0x14, 0xAD,
            0x02, 0xB1, 0x0C, 0xBC, 0x04, 0xB4, 0x0D, 0xBD,
            0xE1, 0xC8, 0xEC, 0xC5, 0xCD, 0xE4, 0xC4, 0xED,
            0xD1, 0xF1, 0xDC, 0xFC, 0xD4, 0xF4, 0xDD, 0xFD,
            0x36, 0x8E, 0x38, 0x82, 0x8B, 0x30, 0x83, 0x39,
            0x96, 0x26, 0x9A, 0x28, 0x93, 0x20, 0x9B, 0x29,
            0x66, 0x4E, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69,
            0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
            0xA6, 0x1E, 0xAA, 0x11, 0x19, 0xA3, 0x10, 0xAB,
            0x06, 0xB6, 0x08, 0xBA, 0x00, 0xB3, 0x09, 0xBB,
            0xE6, 0xCE, 0xEA, 0xC2, 0xCB, 0xE3, 0xC3, 0xEB,
            0xD6, 0xF6, 0xDA, 0xFA, 0xD3, 0xF3, 0xDB, 0xFB,
            0x31, 0x8A, 0x3E, 0x86, 0x8F, 0x37, 0x87, 0x3F,
            0x92, 0x21, 0x9E, 0x2E, 0x97, 0x27, 0x9F, 0x2F,
            0x61, 0x48, 0x6E, 0x46, 0x4F, 0x67, 0x47, 0x6F,
            0x51, 0x71, 0x5E, 0x7E, 0x57, 0x77, 0x5F, 0x7F,
            0xA2, 0x18, 0xAE, 0x16, 0x1F, 0xA7, 0x17, 0xAF,
            0x01, 0xB2, 0x0E, 0xBE, 0x07, 0xB7, 0x0F, 0xBF,
            0xE2, 0xCA, 0xEE, 0xC6, 0xCF, 0xE7, 0xC7, 0xEF,
            0xD2, 0xF2, 0xDE, 0xFE, 0xD7, 0xF7, 0xDF, 0xFF,
        ],
        dtype=np.uint8,
    )
    permutation = np.array(
        [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14],
        dtype=np.int64,
    )

    R_M = 7

    diff_trail = np.array(
        [
            [
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x0A, 0x00,
            ],
        ],
        dtype=np.uint8,
    )

    lin_trail = np.array(
        [
            [
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ],
        ],
        dtype=np.uint8,
    )

    R_D = diff_trail.shape[0] - 1
    R_L = lin_trail.shape[0] - 1

    N = 30
    BATCH_N = 28

    if diff_trail.ndim != 2 or diff_trail.shape[1] != STATE_CELLS:
        raise ValueError("Each row of diff_trail must contain 16 Cell differences.")
    if lin_trail.ndim != 2 or lin_trail.shape[1] != STATE_CELLS:
        raise ValueError("Each row of lin_trail must contain 16 Cell masks.")
    if N < 18:
        raise ValueError("N must be at least 18 for the selected CUDA grid.")

    threads_per_block = 256
    blocks_per_grid = 1024
    total_threads = threads_per_block * blocks_per_grid

    batch_n = min(N, BATCH_N)
    samples_per_batch = 1 << batch_n
    number_of_batches = 1 << (N - batch_n)
    iterations_per_thread = samples_per_batch // total_threads
    actual_samples_per_batch = iterations_per_thread * total_threads
    total_samples = actual_samples_per_batch * number_of_batches

    sbox_gpu = cuda.to_device(sbox)
    permutation_gpu = cuda.to_device(permutation)
    diff_trail_gpu = cuda.to_device(diff_trail)
    lin_trail_gpu = cuda.to_device(lin_trail)
    rng_states = create_xoroshiro128p_states(total_threads, seed=int(time.time()))

    count = np.zeros(4, dtype=np.int64)
    linear_round_count = np.zeros((2, max(1, R_L)), dtype=np.int64)
    start_time = time.time()
    print("BEGIN")
    print(f"Rounds: R_d={R_D}, R_m={R_M}, R_l={R_L}")

    for batch in range(number_of_batches):
        count_gpu = cuda.to_device(np.zeros(4, dtype=np.int64))
        linear_round_count_gpu = cuda.to_device(
            np.zeros((2, max(1, R_L)), dtype=np.int64)
        )
        simulation_kernel[blocks_per_grid, threads_per_block](
            rng_states,
            iterations_per_thread,
            sbox_gpu,
            permutation_gpu,
            R_D,
            R_M,
            R_L,
            diff_trail_gpu,
            lin_trail_gpu,
            count_gpu,
            linear_round_count_gpu,
        )
        count += count_gpu.copy_to_host()
        linear_round_count += linear_round_count_gpu.copy_to_host()
        print(f"Batch {batch + 1}/{number_of_batches} done.")

    accepted = int(count[0])
    print(f"Total pairs: {total_samples}")
    print(f"Counts [accepted, r_1, q0_1, q1_1]: {count}")

    if accepted == 0:
        print(
            "No pair followed the prescribed differential part; "
            "p, r, q, and C are undefined."
        )
    else:
        p_value = accepted / total_samples
        r_value = correlation(int(count[1]), accepted)
        q0_value = correlation(int(count[2]), accepted)
        q1_value = correlation(int(count[3]), accepted)
        c_value = p_value * r_value * q0_value * q1_value

        q0_trail_value = 1.0
        q1_trail_value = 1.0
        for round_index in range(R_L):
            q0_round_value = correlation(
                int(linear_round_count[0, round_index]), accepted
            )
            q1_round_value = correlation(
                int(linear_round_count[1, round_index]), accepted
            )
            q0_trail_value *= q0_round_value
            q1_trail_value *= q1_round_value

            print(
                f"q0[{round_index + 1}] = {q0_round_value:.12e} "
                f"= 2^({log2_value(q0_round_value):.6f})"
            )
            print(
                f"q1[{round_index + 1}] = {q1_round_value:.12e} "
                f"= 2^({log2_value(q1_round_value):.6f})"
            )

        c_trail_value = p_value * r_value * q0_trail_value * q1_trail_value

        p = log2_value(p_value)
        r = log2_value(r_value)
        q0 = log2_value(q0_value)
        q1 = log2_value(q1_value)
        c = log2_value(c_value)

        print(f"p  = {p_value:.12e} = 2^({p:.6f})")
        print(f"r  = {r_value:.12e} = 2^({r:.6f})")
        print(f"q0 (direct E_l) = {q0_value:.12e} = 2^({q0:.6f})")
        print(f"q1 (direct E_l) = {q1_value:.12e} = 2^({q1:.6f})")
        print(f"C  (direct E_l) = {c_value:.12e} = 2^({c:.6f})")

        print(
            f"q0 (trail product) = {q0_trail_value:.12e} "
            f"= 2^({log2_value(q0_trail_value):.6f})"
        )
        print(
            f"q1 (trail product) = {q1_trail_value:.12e} "
            f"= 2^({log2_value(q1_trail_value):.6f})"
        )
        print(
            f"C  (trail product) = {c_trail_value:.12e} "
            f"= 2^({log2_value(c_trail_value):.6f})"
        )

    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("finished")
