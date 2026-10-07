import math

sbox = [12, 0, 15, 10, 2, 11, 9, 5, 8, 3, 13, 7, 1, 14, 6, 4]

ddt = [[0 for _ in range(16)] for _ in range(16)]
for x in range(16):
    for y in range(16):
        ddt[x ^ y][sbox[x] ^ sbox[y]] += 1
for i in range(16):
    for j in range(16):
        ddt[i][j] = float(ddt[i][j]) / 16.0
for i in range(16):
    print(ddt[i])
# print("")
permutation = [1, 2, 11, 6, 3, 0, 9, 4, 7, 10, 13, 14, 5, 8, 15, 12]

trail = [[9], [6], [3, 8], [4, 6], [3, 7, 12], [4, 8, 15], [12, 13, 14], [10, 11], [2], [1]]
x = {"%d_%d" % (r, i): [0.0 for _ in range(16)] for r in range(len(trail)) for i in trail[r]}

file = open("TWINE/Middle_layer/relation.txt", "w")

def sbox_layer(cell):
    temp = [0.0 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            temp[j] += cell[i] * ddt[i][j]
    # print(temp)
    return temp

def xor_layer(a, b):
    temp = [0.0 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            temp[i ^ j] += a[i] * b[j]
    return temp

n = trail[0][0]
for i in range(1, 16):
    x = {"%d_%d" % (r, i): [0.0 for _ in range(16)] for r in range(len(trail)) for i in trail[r]}
    x[f"0_{n}"][i] = 1.0
    file.write("%d\n" % i)
    for r in range(len(trail) - 1):
        for i in range(16):
            next = "%d_%d" % (r + 1, i)
            if next in x:
                if i % 2 == 0:
                    if f"{r}_{permutation[i] - 1}" in x and f"{r}_{permutation[i]}" in x:
                        x[next] = xor_layer(sbox_layer(x[f"{r}_{permutation[i] - 1}"]), x[f"{r}_{permutation[i]}"])
                    elif f"{r}_{permutation[i] - 1}" in x:
                        x[next] = sbox_layer(x[f"{r}_{permutation[i] - 1}"])
                    elif f"{r}_{permutation[i]}" in x:
                        x[next] = x[f"{r}_{permutation[i]}"]
                    else:
                        file.write("wrong\n")
                else:
                    if f"{r}_{permutation[i]}" in x:
                        x[next] = x[f"{r}_{permutation[i]}"]
                    else:
                        file.write("wrong\n")
        file.write(str({k: v for k, v in x.items() if k.startswith(f"{r+1}_")}))
        file.write("\n")
    x[f"0_{n}"][i] = 0.0

    y = x["%d_%d" % (len(trail) - 1, trail[len(trail) - 1][0])]
    file.write("\nFinal output:\n")
    file.write(str(y) + "\n\n")

    output = [[0 for _ in range(2)]  for _ in range(16)]
    for i in range(16):
        i_2 = [int(b) for b in f"{i:04b}"]
        for j in range(16):
            j_2 = [int(b) for b in f"{j:04b}"]
            dot_product = 0
            for k in range(4):
                dot_product ^= i_2[k] * j_2[k]
            output[i][dot_product] += y[j]

    for i in range(16):
        file.write(f"{i:04b}: 0: {output[i][0]:.8f}\t1: {output[i][1]:.8f}\t")
        if output[i][0] != output[i][1]:
            file.write(f"r: {math.log2(float(abs(output[i][0] - output[i][1]) / (output[i][0] + output[i][1]))):.8f}\n")
        else:
            file.write("r: -infinite\n")
    file.write("\n\n")
