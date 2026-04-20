pub fn relu_inplace_simd(data: []f64) void {
    for (data) |*v| if (v.* < 0.0) v.* = 0.0;
}
