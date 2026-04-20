pub fn loss_mse(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0.0;
    for (0..a.len) |i| s += (a[i] - b[i]) * (a[i] - b[i]);
    return s / @as(f64, a.len);
}
