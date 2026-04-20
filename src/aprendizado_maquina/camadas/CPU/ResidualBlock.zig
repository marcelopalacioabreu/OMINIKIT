const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;
const ConvLayer = @import("../CPU/ConvLayer.zig").ConvLayer;
const BatchNormLayer = @import("../CPU/BatchNormLayer.zig").BatchNormLayer;
const ActivationLayer = @import("../CPU/ActivationLayer.zig").ActivationLayer;

pub const ResidualBlock = struct {
    conv1: ?*ConvLayer,
    bn1: ?*BatchNormLayer,
    act: ?*ActivationLayer,
    conv2: ?*ConvLayer,
    bn2: ?*BatchNormLayer,

    pub fn init(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, k: usize) !ResidualBlock {
        var c1 = try ConvLayer.init(ctx, allocator, k);
        var b1 = try BatchNormLayer.init(allocator, 0);
        var a = try ActivationLayer.init(allocator);
        var c2 = try ConvLayer.init(ctx, allocator, k);
        var b2 = try BatchNormLayer.init(allocator, 0);
        return ResidualBlock{ .conv1 = c1, .bn1 = b1, .act = a, .conv2 = c2, .bn2 = b2 };
    }

    pub fn forward(self: *ResidualBlock, ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, input: *tensor) !*tensor {
        var x = try self.conv1.?.forward(ctx, allocator, input);
        x = try self.bn1.?.forward(self.bn1.?, allocator, x, 1e-5);
        x = try self.act.?.relu(self.act.?, allocator, x);
        x = try self.conv2.?.forward(ctx, allocator, x);
        x = try self.bn2.?.forward(self.bn2.?, allocator, x, 1e-5);

        // add skip connection (assume same shape)
        const out = try input.add(allocator, x);
        return out;
    }
};
