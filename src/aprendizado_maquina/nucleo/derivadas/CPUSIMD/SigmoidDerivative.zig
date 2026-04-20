const std = @import("std");
const tensorImpl = @import("../../tensor/TensorImplementacao.zig");
const Backend = tensorImpl.BackendInstance;

pub fn simd_sigmoid_backward(impl_ptr: *Backend, grad: []const f64) void {
    const s = impl_ptr.*;
    const n = s.count;
    const has_scalar = grad.len == 1;
    for (0..n) |i| {
        const upstream = if (has_scalar) grad[0] else grad[i];
        const y = s.data[i];
        const dx = y * (1.0 - y) * upstream;
        s.grad[i] += dx;
    }
}
