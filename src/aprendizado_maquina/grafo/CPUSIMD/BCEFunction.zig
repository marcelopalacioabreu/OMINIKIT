const std = @import("std");
const tensorImpl = @import("../../nucleo/tensor/TensorImplementacao.zig");
const Backend = tensorImpl.BackendInstance;
pub const BCEUserData = @import("../../nucleo/tensor/TensorImplementacao.zig").BCEUserData;

pub fn simd_bce_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*BCEUserData, user.*);
    const pred = ud.pred.*;
    const target = ud.target.*;
    const n = ud.n;

    const has_scalar_upstream = grad.len == 1;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        const eps = 1e-12;
        const pc = if (p < eps) eps else if (p > 1.0 - eps) 1.0 - eps else p;
        const dp = -(t / pc) + ((1.0 - t) / (1.0 - pc));
        pred.grad[i] += upstream * dp;
    }
}
