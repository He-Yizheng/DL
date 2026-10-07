import math

sbox = [[] for _ in range(8)]
sbox[0] = [14, 9, 15, 0, 13, 4, 10, 11, 1, 2, 8, 3, 7, 6, 12, 5] # S0
sbox[1] = [4, 11, 14, 9, 15, 13, 0, 10, 7, 12, 5, 6, 2, 8, 1, 3] # S1
sbox[2] = [1, 14, 7, 12, 15, 13, 0, 6, 11, 5, 9, 3, 2, 4, 8, 10] # S2
sbox[3] = [7, 6, 8, 11, 0, 15, 3, 14, 9, 10, 12, 13, 5, 2, 4, 1] # S3
sbox[4] = [14, 5, 15, 0, 7, 2, 12, 13, 1, 8, 4, 9, 11, 10, 6, 3] # S4
sbox[5] = [2, 13, 11, 12, 15, 14, 0, 9, 7, 10, 6, 3, 1, 8, 4, 5] # S5
sbox[6] = [11, 9, 4, 14, 0, 15, 10, 13, 6, 12, 5, 7, 3, 8, 1, 2] # S6
sbox[7] = [13, 10, 15, 0, 14, 4, 9, 11, 2, 1, 8, 3, 7, 5, 12, 6] # S7

ddt = [[[0 for _ in range(16)] for _ in range(16)] for _ in range(8)]
for r in range(8):
    for x in range(16):
        for y in range(16):
            ddt[r][x ^ y][sbox[r][x] ^ sbox[r][y]] += 1
for r in range(8):   
    for i in range(16):
        for j in range(16):
            if ddt[r][i][j] == 2:
                ddt[r][i][j] = 0.125
            elif ddt[r][i][j] == 4:
                ddt[r][i][j] = 0.25
            elif ddt[r][i][j] == 16:
                ddt[r][i][j] = 1.0
            
def sbox(a):
    temp = [0.0 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            temp[j] += ddt[7 - a[0]][i][j] * a[1][i]
    return temp

def xor(a, b):
    temp = [0.0 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            temp[i ^ j] += a[i] * b[j]
    return temp

permutation = [1, 3, 0, 2, 5, 7, 4, 6]

a = [[8], [6], [7, 14], [4, 5], [6, 12, 13], [2, 3, 14], [1, 4, 10], [0, 12], [2], [10]]
x = {"%d_%d" % (r, i): [0.0 for _ in range(16)] for r in range(len(a)) for i in a[r]}
x["0_8"][3] = 1.0

for r in range(len(a) - 1):
    for i in range(8):
        key_next = f"{r+1}_{i+8}"
        key_prev = f"{r}_{i}"
        if key_next in x:
            x[key_next] = x[key_prev]
        key_next_i = f"{r+1}_{i}"
        key_perm = f"{r}_{permutation[i]}"
        key_xor = f"{r}_{(i+2)%8+8}"
        if key_next_i in x:
            if key_perm in x and key_xor not in x:
                print(1)
                x[key_next_i] = sbox([permutation[i], x[key_perm]])
            elif key_xor in x and key_perm in x:
                print(2)
                x[key_next_i] = xor(sbox([permutation[i], x[key_perm]]), x[key_xor])
            elif key_xor in x and key_perm not in x:
                print(3)
                x[key_next_i] = x[key_xor]
            else:
                print(4)
    print({k: v for k, v in x.items() if k.startswith(f"{r+1}_")})

x = x["%d_%d" % (len(a) - 1, a[len(a) - 1][0])]
print("\nFinal output:")
print(x)
print("")
# 0001
a = x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] + x[10] - x[11] + x[12] - x[13] + x[14] - x[15]
print("\n0001: %f" % abs(a))
if a != 0:
    print("r:%f" % math.log2(abs(a)))
else:
    print("-inf")

# # 0010
b = x[0] + x[1] - x[2] - x[3] + x[4] + x[5] - x[6] - x[7] + x[8] + x[9] - x[10] - x[11] + x[12] + x[13] - x[14] - x[15]
print("\n0010: %f" % abs(b))
if b != 0:
    print("r:%f" % math.log2(abs(b)))
else:
    print("-inf")

# 0100
c = x[0] + x[1] + x[2] + x[3] - x[4] - x[5] - x[6] - x[7] + x[8] + x[9] + x[10] + x[11] - x[12] - x[13] - x[14] - x[15]
print("\n0100: %f" % abs(c))
if c != 0:
    print("r:%f" % math.log2(abs(c)))
else:
    print("-inf")

# 1000
d = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] - x[8] - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15]
print("\n1000: %f" % abs(d))
if d != 0:
    print("r:%f" % math.log2(abs(d)))
else:
    print("-inf")
