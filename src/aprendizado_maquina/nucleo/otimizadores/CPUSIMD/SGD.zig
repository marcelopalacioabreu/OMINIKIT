const std = @import("std");
const tensorImpl = @import("../../tensor/TensorImplementacao.zig");

const Backend = tensorImpl.BackendInstance;

pub fn simd_sgd_update(impl_ptr: *Backend, lr: f64) void {
    const s = impl_ptr.*;
    for (0..s.count) |i| {
        const g = s.grad[i];
        s.data[i] = s.data[i] - lr * g;
        s.grad[i] = 0.0;
    }
}
