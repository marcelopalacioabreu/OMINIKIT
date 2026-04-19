const std = @import("std");
const tensor_mod = @import("Tensor.zig");
const computacao = @import("../../../computacao/ComputacaoContexto.zig");

pub const FabricaTensor = struct {
    ctx: *computacao.ComputacaoContexto,
    allocator: *std.mem.Allocator,

    pub fn init(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator) FabricaTensor {
        return FabricaTensor{
            .ctx = ctx,
            .allocator = allocator,
        };
    }

    pub fn criar(self: *FabricaTensor, shape: []const usize) !*tensor_mod.Tensor {
        return tensor_mod.Tensor.init(self.ctx, self.allocator, shape);
    }

    pub fn fromArray(self: *FabricaTensor, shape: []const usize, data: []const f64) !*tensor_mod.Tensor {
        return tensor_mod.Tensor.fromArray(self.ctx, self.allocator, shape, data);
    }
};
