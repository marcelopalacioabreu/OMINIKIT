const std = @import("std");
const tensorImpl = @import("../../tensor/TensorImplementacao.zig");
const Backend = tensorImpl.BackendInstance;

pub fn relu_backward(impl_ptr: *Backend, grad: []const f64) void {
    const s = impl_ptr.*;
    const n = s.count;
    const has_scalar = grad.len == 1;
    for (0..n) |i| {
        const upstream = if (has_scalar) grad[0] else grad[i];
        const v = s.data[i];
        if (v > 0.0) s.grad[i] += upstream; else s.grad[i] += 0.0;
    }
}
