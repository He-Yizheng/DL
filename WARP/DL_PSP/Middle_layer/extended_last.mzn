
int: R;
int: INPUT_CELL;
int: OUTPUT_COUNT;
array[0..OUTPUT_COUNT-1] of int: OUTPUT_CELLS;

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

constraint forall(r in 0..(R - 1), l in 0..15)(
    y[r + 1, l * 2] = (y[r, permutation[l * 2]] \/ y[r, permutation[l * 2 + 1]])
    /\
    y[r + 1, l * 2 + 1] = y[r, permutation[l * 2 + 1]]
);

constraint forall(l in 0..31)(
    if l == INPUT_CELL then x[0, l] = true
    else x[0, l] = false
    endif
);

constraint forall(l in 0..31)(
    if exists(i in 0..OUTPUT_COUNT-1)(l == OUTPUT_CELLS[i]) then y[0, l] = true
    else y[0, l] = false
    endif
);

solve satisfy;
