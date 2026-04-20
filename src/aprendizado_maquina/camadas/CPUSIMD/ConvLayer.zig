const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;

pub const ConvLayer = struct {
    weight: ?*tensor,
    k: usize,

    pub fn init(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, k_in: usize) !*ConvLayer {
        var obj = try allocator.create(ConvLayer);
        const shape = [_]usize{ k_in, k_in };
        obj.weight = try tensor.init(ctx, allocator, shape[0..shape.len]);
        obj.k = k_in;
        return obj;
    }

    pub fn forward(self: *ConvLayer, ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, input: *tensor) !*tensor {
        if (self.weight == null) return error.InvalidArgument;
        return input.conv(allocator, self.weight.?);
    }
};
