const std = @import("std");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;
const computacao = @import("../../../computacao/mod.zig");

pub const ImageLoadError = error{ FileNotFound, DecodeFailed, OutOfMemory };

pub const GrayImage = struct {
    buf: []u8,
    width: usize,
    height: usize,
};

pub fn carregarComoGray(allocator: *std.mem.Allocator, path: []const u8) !GrayImage {
    // Prefer the vendored zignal adapter if present
    const ZignalAdapter = @import("ZignalAdapter.zig");
    const res = ZignalAdapter.loadAsGray(allocator, path) catch |e| {
        return e;
    };
    return res;
}

pub fn transformarBufferEmTensor(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, buf: []const u8, width: usize, height: usize) !*tensor {
    const ft = @import("../../nucleo/tensor/FabricaTensor.zig").FabricaTensor.init(ctx, allocator);
    const t = try ft.criar(&[_]usize{ height, width });
    // normalize u8 [0..255] to f64 [0..1]
    for (0..t.size) |i| {
        const v = @as(f64, buf[i]) / 255.0;
        t.set(i, v);
    }
    return t;
}

pub fn transformarEmTensor(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, width: usize, height: usize) !*tensor {
    // legacy stub kept for compatibility: returns zero tensor
    const ft = @import("../../nucleo/tensor/FabricaTensor.zig").FabricaTensor.init(ctx, allocator);
    const t = try ft.criar(&[_]usize{ height, width });
    for (0..t.size) |i| t.set(i, 0.0);
    return t;
}
