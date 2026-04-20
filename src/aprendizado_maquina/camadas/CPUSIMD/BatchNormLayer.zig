const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;

pub const BatchNormLayer = struct {
    pub fn init(allocator: *std.mem.Allocator, channels: usize) !*BatchNormLayer {
        var obj = try allocator.create(BatchNormLayer);
        return obj;
    }

    pub fn forward(self: *BatchNormLayer, allocator: *std.mem.Allocator, input: *tensor, epsilon: f64) !*tensor {
        return input.batchnorm(allocator, epsilon);
    }
};
