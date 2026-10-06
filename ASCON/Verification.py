import math
import time
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

cuda.select_device(0)

@cuda.jit(device=True)
def sbox_function(x, x_sb, sbox):
    for i in range(64):
        temp = sbox[x[i, 0] * 16 + x[i, 1] * 8 + x[i, 2] * 4 + x[i, 3] * 2 + x[i, 4]]
        x_sb[i, 0] = temp >> 4 & 1
        x_sb[i, 1] = temp >> 3 & 1
        x_sb[i, 2] = temp >> 2 & 1
        x_sb[i, 3] = temp >> 1 & 1
        x_sb[i, 4] = temp & 1

@cuda.jit(device=True)
def linear_function(x_sb, x):
    for i in range(64):
        x[i, 0] = x_sb[i, 0] ^ x_sb[(i + 45) % 64, 0] ^ x_sb[(i + 36) % 64, 0]
        x[i, 1] = x_sb[i, 1] ^ x_sb[(i + 3) % 64, 1] ^ x_sb[(i + 25) % 64, 1]
        x[i, 2] = x_sb[i, 2] ^ x_sb[(i + 63) % 64, 2] ^ x_sb[(i + 58) % 64, 2]
        x[i, 3] = x_sb[i, 3] ^ x_sb[(i + 54) % 64, 3] ^ x_sb[(i + 47) % 64, 3]
        x[i, 4] = x_sb[i, 4] ^ x_sb[(i + 57) % 64, 4] ^ x_sb[(i + 23) % 64, 4]


@cuda.jit
def simulation_kernel(rng_states, iterations_per_thread, sbox, input_diff0, input_diff1, output_lin0, output_lin1, count0, count1):
    thread_id = cuda.grid(1)
    input0 = cuda.local.array((64, 5), dtype=np.uint8)
    input1 = cuda.local.array((64, 5), dtype=np.uint8)
    input0_sb = cuda.local.array((64, 5), dtype=np.uint8)
    input1_sb = cuda.local.array((64, 5), dtype=np.uint8)
    
    for _ in range(iterations_per_thread):
        for i in range(64):
            for j in range(5):
                input0[i, j] = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * 2)
                input1[i, j] = input0[i, j] ^ input_diff0[i, j]
        sbox_function(input0, input0_sb, sbox)
        sbox_function(input1, input1_sb, sbox)
        linear_function(input0_sb, input0)
        linear_function(input1_sb, input1)
        flag = True
        for i in range(64):
            for j in range(5):
                if input0[i, j] ^ input1[i, j] != input_diff1[i, j]:
                    flag = False
                    break
        if flag:
            cuda.atomic.add(count0, 0, 1)
            for _ in range(round):
                sbox_function(input0, input0_sb, sbox)
                sbox_function(input1, input1_sb, sbox)
                linear_function(input0_sb, input0)
                linear_function(input1_sb, input1)
            output_lin = 0
            for i in range(64):
                for j in range(5):
                    output_lin ^= (input0[i, j] ^ input1[i, j]) * output_lin0[i, j]
            if output_lin == 1:
                cuda.atomic.add(count0, 1, 1)
            sbox_function(input0, input0_sb, sbox)
            sbox_function(input1, input1_sb, sbox)
            for i in range(64):
                output_lin = 0
                for j in range(5):
                    output_lin ^= output_lin0[i, j] * input0[i, j]
                    output_lin ^= output_lin1[i, j] * input0_sb[i, j]
                if output_lin == 1:
                    cuda.atomic.add(count1, (0, i), 1)
            for i in range(64):
                output_lin = 0
                for j in range(5):
                    output_lin ^= output_lin0[i, j] * input1[i, j]
                    output_lin ^= output_lin1[i, j] * input1_sb[i, j]
                if output_lin == 1:
                    cuda.atomic.add(count1, (1, i), 1)                
        
        
        
if __name__ == "__main__":
    sbox = np.array([
        4, 11, 31, 20, 26, 21, 9, 2, 27, 5, 8, 18, 29, 3, 6, 28, 
        30, 19, 7, 14, 0, 13, 17, 24, 16, 12, 1, 25, 22, 10, 15, 23
    ], dtype=np.int64)
    round = 3
    n = 40

    round_times = n - 28
    times = 1 << 28
    threads_per_block = 256
    blocks_per_grid = 1024
    total_threads = threads_per_block * blocks_per_grid
    iterations_per_thread = times // total_threads
    sbox_gpu = cuda.to_device(sbox)
    rng_states = create_xoroshiro128p_states(total_threads, seed=int(time.time()))

    diff0 = ["" for _ in range(5)]
    diff0[0] = "0000000000100000000000000000000000000000000000000000000000000000"
    diff0[1] = "0000000000000000000000000000000000000000000000000000000000000000"
    diff0[2] = "0000000000000000000000000000000000000000000000000000000000000000"
    diff0[3] = "0000000000100000000000000000000000000000000000000000000000000000"
    diff0[4] = "0000000000100000000000000000000000000000000000000000000000000000"
    for i in range(5):
        diff0[i] = [int(b) for b in diff0[i]]
        # print(diff0[i])

    diff1 = ["" for _ in range(5)]
    diff1[0] = "0000000000000000000000000000000000000000000000000000000000000000"
    diff1[1] = "0000000000000000000000000000000000000000000000000000000000000000"
    diff1[2] = "0000000000000000000000000000000000000000000000000000000000000000"
    diff1[3] = "0000000000100000000010000001000000000000000000000000000000000000"
    diff1[4] = "0000000000000000000000000000000000000000000000000000000000000000"
    for i in range(5):
        diff1[i] = [int(b) for b in diff1[i]]
        # print(diff1[i])

    lin0 = ["" for _ in range(5)]
    lin0[0] = "0000000000000000000000000000000000000000000000000000000000000000"
    lin0[1] = "0000000000000000000000000000000000000000000000000000000000000000"
    lin0[2] = "0000000000000000000000000000000000000000000000000000000000000000"
    lin0[3] = "1000000000000000000000000000000000000000000000000000000000000000"
    lin0[4] = "0000000000000000000000000000000000000000000000000000000000000000"
    for i in range(5):
        lin0[i] = [int(b) for b in lin0[i]]

    lin1 = ["" for _ in range(5)]
    lin1[0] = "0000000000000000000000000000000000000000000000000000000000000000"
    lin1[1] = "0000000000000000000000000000000000000000000000000000000000000000"
    lin1[2] = "1000000000000000000000000000000000000000000000000000000000000000"
    lin1[3] = "1000000000000000000000000000000000000000000000000000000000000000"
    lin1[4] = "0000000000000000000000000000000000000000000000000000000000000000"
    for i in range(5):
        lin1[i] = [int(b) for b in lin1[i]]

    time_s = time.time()
    print("BEGIN")
    time0 = time.time()
    input_diff0 = np.zeros((64, 5), dtype=np.uint8)
    input_diff1 = np.zeros((64, 5), dtype=np.uint8)
    for i in range(64):
        for j in range(5):
            input_diff0[i, j] = diff0[j][i]
            input_diff1[i, j] = diff1[j][i]
    input_diff0_gpu = cuda.to_device(input_diff0)
    input_diff1_gpu = cuda.to_device(input_diff1)

    output_lin0 = np.zeros((64, 5), dtype=np.uint8)
    output_lin1 = np.zeros((64, 5), dtype=np.uint8)
    for i in range(64):
        for j in range(5):
            output_lin0[i, j] = lin0[j][i]
            output_lin1[i, j] = lin1[j][i]
    output_lin0_gpu = cuda.to_device(output_lin0)
    output_lin1_gpu = cuda.to_device(output_lin1)

    count0 = np.zeros(2, dtype=np.int64)
    count1 = np.zeros((2, 64), dtype=np.int64)
    
    for rt in range(1 << round_times):
        count0_gpu = cuda.to_device(np.zeros(2, dtype=np.int64))
        count1_gpu = cuda.to_device(np.zeros((2, 64), dtype=np.int64))

        simulation_kernel[blocks_per_grid, threads_per_block](
            rng_states, iterations_per_thread, sbox_gpu, input_diff0_gpu, input_diff1_gpu, output_lin0_gpu, output_lin1_gpu, count0_gpu, count1_gpu
        )

        count0_round = count0_gpu.copy_to_host()
        count1_round = count1_gpu.copy_to_host()

        for i in range(2):
            count0[i] += count0_round[i]
        for i in range(2):
            for j in range(64):
                count1[i, j] += count1_round[i, j]

        print(f"Round Time {rt} done.")
    
    print(f" Total: {1 << n}")
    print(f"Count0: {count0}")
    print(f"Count1: {count1}\n")

    p = math.log2(float(count0[0] / (1 << n)))
    print(f" p: 2^{p:.6f}")
    r = math.log2(float((abs(count0[1] * 2 - count0[0])) / count0[0]))
    print(f" r: 2^{r:.6f}")

    q0 = 0.0
    q1 = 0.0
    for i in range(64):
        q0 += math.log2(float(abs((count1[0, i]) * 2 - count0[0])) / count0[0])
        q1 += math.log2(float(abs((count1[1, i]) * 2 - count0[0])) / count0[0])
    print(f" q0: 2^{q0:.6f}")
    print(f" q1: 2^{q1:.6f}\n")
    
    cr = p + r + q0 + q1
    print(f"cr: 2^{cr:.6f}\n")

    print(f"Total time: {time.time() - time_s} s")
    print("finished")
