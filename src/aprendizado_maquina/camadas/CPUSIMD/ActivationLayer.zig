const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;

pub const ActivationLayer = struct {
    pub fn init(allocator: *std.mem.Allocator) !*ActivationLayer {
        var obj = try allocator.create(ActivationLayer);
        return obj;
    }

    pub fn relu(self: *ActivationLayer, allocator: *std.mem.Allocator, input: *tensor) !*tensor {
        const out = try tensor.init_with_type(input.tipo, allocator, input.shape);
        for (0..input.size) |i| {
            const v = input.get(i);
            if (v > 0.0) out.set(i, v) else out.set(i, 0.0);
        }
        // attach userdata and backward for autograd
        const impl = out.impl_ptr;
        const tensorImpl = @import("../../nucleo/tensor/TensorImplementacao.zig");
        const cpusimd = @import("../../nucleo/tensor/TensorCPUSIMD.zig");
        const ud = try allocator.create(tensorImpl.AnyUserData);
        ud.* = .{ .relu = .{ .input = input.impl_ptr } };
        impl.user = ud;
        switch (input.tipo) {
            .CPUSIMD => out.grad_fn = &cpusimd.simd_relu_backward,
            .CPU => out.grad_fn = &cpusimd.simd_relu_backward,
            else => out.grad_fn = &cpusimd.simd_relu_backward,
        }
        return out;
    }
};
