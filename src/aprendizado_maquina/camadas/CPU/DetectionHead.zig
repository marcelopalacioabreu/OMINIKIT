const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;
const ConvLayer = @import("../CPU/ConvLayer.zig").ConvLayer;
const ActivationLayer = @import("../CPU/ActivationLayer.zig").ActivationLayer;

pub const DetectionHead = struct {
    conv1: ?*ConvLayer,
    act: ?*ActivationLayer,
    conv2: ?*ConvLayer,

    pub fn init(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, k: usize) !DetectionHead {
        var c1 = try ConvLayer.init(ctx, allocator, k);
        var a = try ActivationLayer.init(allocator);
        var c2 = try ConvLayer.init(ctx, allocator, k);
        return DetectionHead{ .conv1 = c1, .act = a, .conv2 = c2 };
    }

    pub fn forward(self: *DetectionHead, ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, input: *tensor) !*tensor {
        var x = try self.conv1.?.forward(ctx, allocator, input);
        x = try self.act.?.relu(self.act.?, allocator, x);
        x = try self.conv2.?.forward(ctx, allocator, x);
        return x;
    }
};
