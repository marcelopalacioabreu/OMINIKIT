const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const CPUConv = @import("CPU/ConvLayer.zig").ConvLayer;
const SIMDConv = @import("CPUSIMD/ConvLayer.zig").ConvLayer;

pub const FabricaCamadas = struct {
    pub const ConvHandle = union(enum) { CPU: *CPUConv, CPUSIMD: *SIMDConv };

    pub fn createConvLayer(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, k: usize) !ConvHandle {
        switch (ctx.tipo) {
            .CPU => {
                const l = try CPUConv.init(ctx, allocator, k);
                return ConvHandle{ .CPU = l };
            },
            .CPUSIMD => {
                const l = try SIMDConv.init(ctx, allocator, k);
                return ConvHandle{ .CPUSIMD = l };
            },
            else => {
                const l = try CPUConv.init(ctx, allocator, k);
                return ConvHandle{ .CPU = l };
            },
        }
    }
};
