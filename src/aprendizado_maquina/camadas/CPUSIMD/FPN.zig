const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;
const ConvLayer = @import("../CPUSIMD/ConvLayer.zig").ConvLayer;

pub const FPN = struct {
    lateral: ?*ConvLayer,
    output: ?*ConvLayer,

    pub fn init(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, k: usize) !FPN {
        var lat = try ConvLayer.init(ctx, allocator, k);
        var out = try ConvLayer.init(ctx, allocator, k);
        return FPN{ .lateral = lat, .output = out };
    }

    pub fn forward(self: *FPN, ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, c3: *tensor, c4: *tensor, c5: *tensor) ![*] ?*tensor {
        const upsample = fn (allocator: *std.mem.Allocator, t: *tensor, factor: usize) !*tensor {
            const h = t.shape[0];
            const w = t.shape[1];
            const nh = h * factor;
            const nw = w * factor;
            const res = try tensor.init_with_type(t.tipo, allocator, &[_]usize{ nh, nw });
            for (0..nh) |i| for (0..nw) |j| {
                const src_i = i / factor;
                const src_j = j / factor;
                res.set(i * nw + j, t.get(src_i * w + src_j));
            }
            return res;
        };

        var p5 = try self.output.?.forward(ctx, allocator, c5);
        var p5_up = try upsample(allocator, p5, 2);
        var p4 = try c4.add(allocator, p5_up);
        var p4_up = try upsample(allocator, p4, 2);
        var p3 = try c3.add(allocator, p4_up);

        var arr = try allocator.alloc(?*tensor, 3);
        arr[0] = p3;
        arr[1] = p4;
        arr[2] = p5;
        return arr;
    }
};
