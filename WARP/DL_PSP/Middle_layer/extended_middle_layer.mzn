
int: R;
int: INPUT_CELL;
int: DEFECT_CELL;

array[0..R, 0..31] of var bool: x;
array[0..R, 0..31] of var bool: y;

array[0..31] of int: permutation = array1d(0..31, [
    31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25, 4, 23, 10,
    15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26
]);

constraint forall(r in 0..(R - 1), l in 0..15)(
    x[r + 1, permutation[l * 2]] = x[r, l * 2]
    /\
    x[r + 1, permutation[l * 2 + 1]] = (x[r, l * 2] \/ x[r, l * 2 + 1])
);

constraint forall(r in 2..(R - 1), l in 0..15)(
    y[r + 1, l * 2] = (y[r, permutation[l * 2]] \/ y[r, permutation[l * 2 + 1]])
    /\
    y[r + 1, l * 2 + 1] = y[r, permutation[l * 2 + 1]]
);

constraint forall(r in 0..1, l in 0..15)(
    y[r, permutation[l * 2]] = y[r + 1, l * 2]
    /\
    y[r, permutation[l * 2 + 1]] = (y[r + 1, l * 2] \/ y[r + 1, l * 2 + 1])
);

constraint forall(l in 0..31)(
    if l == INPUT_CELL then x[0, l] = true
    else x[0, l] = false
    endif
);

constraint forall(l in 0..31)(
    if l == DEFECT_CELL then y[2, l] = true
    else y[2, l] = false
    endif
);

solve satisfy;
