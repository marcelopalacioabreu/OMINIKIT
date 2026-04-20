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
        return out;
    }
};
