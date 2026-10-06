int: R;

array[0..R, 0..15] of var bool: x;

array[0..R, 0..15] of var bool: y;

constraint forall(r in 0..(R - 1), l in 0..3)(
    x[r + 1, l] = (x[r, l] \/ x[r, (l + 2) mod 4 + 8] \/ x[r, (l + 1) mod 4 + 12])
    /\
    x[r + 1, l + 4] = x[r, l]
    /\
    x[r + 1, l + 8] = (x[r, (l + 3) mod 4 + 4] \/ x[r, (l + 2) mod 4 + 8])
    /\
    x[r + 1, l + 12] = (x[r, l] \/ x[r, (l + 2) mod 4 + 8])
);

constraint forall(r in 0..(R - 1), l in 0..3)(
    y[r + 1, l] = (y[r, l] \/ y[r, l + 4] \/ y[r, l + 12])
    /\
    y[r + 1, l + 4] = y[r, (l + 1) mod 4 + 8]
    /\
    y[r + 1, l + 8] = (y[r, (l + 2) mod 4] \/ y[r, (l + 2) mod 4 + 8] \/ y[r, (l + 3) mod 4 + 12])
    /\
    y[r + 1, l + 12] = (y[r, (l + 3) mod 4])
);

constraint forall(l in 0..15)(
    if l == 14 then x[0, l] = true
    % elseif l == 6 then x[0, l] = true
    else x[0, l] = false
    endif
);

constraint forall(l in 0..15)(
    if l == 12 then y[0, l] = true
    % elseif l == 12 then y[0, l] = true
    else y[0, l] = false
    endif
);

solve satisfy;