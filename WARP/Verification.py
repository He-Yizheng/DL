import math
import time
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

cuda.select_device(0)

@cuda.jit(device=True)
def round_function(x, x_s, sbox, permutation):
    for i in range(16):
        x_s[permutation[i * 2]] = x[i * 2]
        x_s[permutation[i * 2 + 1]] = sbox[x[i * 2]] ^ x[i * 2 + 1]
    for i in range(32):
        x[i] = x_s[i]

@cuda.jit
def simulation_kernel(rng_states, iterations_per_thread, sbox, permutation, round, diff, lin, count_diff, count_dl, count_lin):
    thread_id = cuda.grid(1)
    input0 = cuda.local.array(32, dtype=np.uint8)
    input0_s = cuda.local.array(32, dtype=np.uint8)
    input1 = cuda.local.array(32, dtype=np.uint8)
    input1_s = cuda.local.array(32, dtype=np.uint8)
    output = cuda.local.array(32, dtype=np.uint8)

    for _ in range(iterations_per_thread):
        for i in range(32):
            input0[i] = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * 16)
            input1[i] = input0[i] ^ diff[0, i]
        flag = True
        for r in range(round[0]):
            round_function(input0, input0_s, sbox, permutation)
            round_function(input1, input1_s, sbox, permutation)
            for i in range(32):
                if input0[i] ^ input1[i] != diff[r + 1, i]:
                    flag = False
            if not flag:
                cuda.atomic.add(count_diff, (r, 0), 1)
                break
            else:
                cuda.atomic.add(count_diff, (r, 1), 1)
        if flag:
            for _ in range(round[1]):
                round_function(input0, input0_s, sbox, permutation)
                round_function(input1, input1_s, sbox, permutation)
            for i in range(32):
                output[i] = input0[i] ^ input1[i]
            dl_output = 0
            for i in range(32):
                for j in range(4):
                    dl_output ^= (output[i] >> j & 1) * (lin[0, 0, i] >> j & 1)
            cuda.atomic.add(count_dl, dl_output, 1)
            for r in range(round[2]):
                for i in range(16):
                    output_lin = 0
                    for j in range(4):
                        output_lin ^= (lin[r, 0, i * 2] >> j & 1) * (input0[i * 2] >> j & 1)
                        output_lin ^= (lin[r, 1, i * 2] >> j & 1) * (sbox[input0[i * 2]] >> j & 1)
                    cuda.atomic.add(count_lin, (r, i, 0, output_lin), 1)
                    output_lin = 0
                    for j in range(4):
                        output_lin ^= (lin[r, 0, i * 2] >> j & 1) * (input1[i * 2] >> j & 1)
                        output_lin ^= (lin[r, 1, i * 2] >> j & 1) * (sbox[input1[i * 2]] >> j & 1)
                    cuda.atomic.add(count_lin, (r, i, 1, output_lin), 1)
                round_function(input0, input0_s, sbox, permutation)
                round_function(input1, input1_s, sbox, permutation)



if __name__ == "__main__":
    sbox = np.array([
        12, 10, 13, 3, 14, 11, 15, 7, 8, 9, 1, 5, 0, 2, 4, 6
    ], dtype=np.uint8)
    permutation = np.array([
        31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10,
        15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26
    ])

    diff_round = 1
    dl_round = 11
    lin_round = 1
    n = 40

    times = 1 << n
    threads_per_block = 256
    blocks_per_grid = 1024
    total_threads = threads_per_block * blocks_per_grid
    iterations_per_thread = times // total_threads
    sbox_gpu = cuda.to_device(sbox)
    permutation_gpu = cuda.to_device(permutation)
    round_gpu = cuda.to_device(np.array([diff_round, dl_round, lin_round], dtype=np.uint8))
    rng_states = create_xoroshiro128p_states(total_threads, seed=int(time.time()))

    diff = [[] for _ in range(diff_round + 1)]
    # diff[3] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 15, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]
    # diff[2] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 10, 5, 0, 0, 0, 0, 0, 0]
    diff[1] = [0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    diff[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    diff = np.array(diff, dtype=np.uint8)
    diff_gpu = cuda.to_device(diff)

    lin = [[] for _ in range(lin_round + 1)]
    lin[0] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    lin[1] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0]
    ]
    # lin[2] = [
    #     [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
    # ]
    # lin[3] = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    lin = np.array(lin, dtype=np.uint8)
    lin_gpu = cuda.to_device(lin)

    time_start = time.time()
    print("BEGIN")

    file = open(f"WARP/GPU_verify.txt", "a")
    file.write(f"DIFF rounds: {diff_round}\n")
    file.write(f"DL rounds: {dl_round}\n")
    file.write(f"LIN rounds: {lin_round}\n\n")
    file.write(f"n: {1 << n}\n")
    
    count_diff_gpu = cuda.to_device(np.zeros((diff_round, 2), dtype=np.int64))
    count_dl_gpu = cuda.to_device(np.zeros(2, dtype=np.int64))
    count_lin_gpu = cuda.to_device(np.zeros((lin_round, 16, 2, 2), dtype=np.int64))

    simulation_kernel[blocks_per_grid, threads_per_block](
        rng_states,
        iterations_per_thread,
        sbox_gpu,
        permutation_gpu,
        round_gpu,
        diff_gpu,
        lin_gpu,
        count_diff_gpu,
        count_dl_gpu,
        count_lin_gpu
    )

    count_diff = count_diff_gpu.copy_to_host()
    count_dl = count_dl_gpu.copy_to_host()
    count_lin = count_lin_gpu.copy_to_host()
    time_end = time.time()



    file.write(f"Time: {time_end - time_start} seconds\n\n")
    file.write(f"DIFF:\n{diff}\n")
    file.write(f"LIN:\n{lin}\n\n")

    file.write(f"DIFF Count:\n{count_diff}\n")
    file.write(f"DL Count:\n{count_dl}\n")
    file.write(f"LIN Count:\n{count_lin}\n\n")

    p = 0.0
    for R in range(diff_round):
        p += math.log2(float(count_diff[R, 1] / (count_diff[R, 0] + count_diff[R, 1])))
    file.write("Differential Probability: 2^%.6f\n" % p)
    r = math.log2(float((abs(count_dl[1] * 2 - (count_dl[0] + count_dl[1]))) / (count_dl[0] + count_dl[1])))
    file.write("Differential-Linear Correlation: 2^%.6f\n" % r)
    q = 0.0
    for R in range(lin_round):
        for i in range(16):
            c0 = count_lin[R, i, 0, 0] + count_lin[R, i, 1, 0]
            c1 = count_lin[R, i, 0, 1] + count_lin[R, i, 1, 1]
            q += math.log2(float((abs(c0 - c1)) / (c0 + c1)))
    file.write("Linear Probability: 2^%.6f\n" % q)

    cr = p + r + q * 2
    file.write("Correlation: 2^%.6f\n\n\n" % cr)
    file.close()
