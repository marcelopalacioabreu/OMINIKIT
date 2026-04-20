const std = @import("std");
const computacao = @import("../../computacao/ComputacaoContexto.zig");
const MatMulSimd = @import("CPUSIMD/MatMulFunction.zig").MatMulFunction;
const ConvSimd = @import("CPUSIMD/ConvFunction.zig").ConvFunction;
const BNSimd = @import("CPUSIMD/BNFunction.zig").BNFunction;
const BCECpu = @import("CPU/BCEFunction.zig");
const BCEsimd = @import("CPUSIMD/BCEFunction.zig");
const SL1Cpu = @import("CPU/SmoothL1Function.zig");
const SL1simd = @import("CPUSIMD/SmoothL1Function.zig");

pub const FabricaFuncoesRetropropagacao = struct {
    pub fn create_matmul(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, user: ?*u8) !*MatMulSimd {
        switch (ctx.tipo) {
            .CPU => {
                const M = @import("CPU/MatMulFunction.zig").MatMulFunction;
                var f = try allocator.create(M);
                f.* = M.init(allocator, user);
                return f;
            },
            .CPUSIMD => {
                var f = try allocator.create(MatMulSimd);
                f.* = MatMulSimd.init(allocator, user);
                return f;
            },
            else => {
                var f = try allocator.create(MatMulSimd);
                f.* = MatMulSimd.init(allocator, user);
                return f;
            },
        }
    }

    pub fn create_conv(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, user: ?*u8) !*ConvSimd {
        switch (ctx.tipo) {
            .CPU => {
                const C = @import("CPU/ConvFunction.zig").ConvFunction;
                var f = try allocator.create(C);
                f.* = C.init(allocator, user);
                return f;
            },
            .CPUSIMD => {
                var f = try allocator.create(ConvSimd);
                f.* = ConvSimd.init(allocator, user);
                return f;
            },
            else => {
                var f = try allocator.create(ConvSimd);
                f.* = ConvSimd.init(allocator, user);
                return f;
            },
        }
    }

    pub fn create_bn(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, user: ?*u8) !*BNSimd {
        switch (ctx.tipo) {
            .CPU => {
                const B = @import("CPU/BNFunction.zig").BNFunction;
                var f = try allocator.create(B);
                f.* = B.init(allocator, user);
                return f;
            },
            .CPUSIMD => {
                var f = try allocator.create(BNSimd);
                f.* = BNSimd.init(allocator, user);
                return f;
            },
            else => {
                var f = try allocator.create(BNSimd);
                f.* = BNSimd.init(allocator, user);
                return f;
            },
        }
    }

    pub fn create_bce(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, user: ?*u8) !*anyopaque {
        // choose CPU or CPUSIMD BCE userdata structures
        switch (ctx.tipo) {
            .CPU => {
                var ud = try allocator.create(BCECpu.BCEUserData);
                ud.* = .{ .pred = null, .target = null, .n = 0 };
                return ud;
            },
            .CPUSIMD => {
                var ud = try allocator.create(BCEsimd.BCEUserData);
                ud.* = .{ .pred = null, .target = null, .n = 0 };
                return ud;
            },
            else => {
                var ud = try allocator.create(BCEsimd.BCEUserData);
                ud.* = .{ .pred = null, .target = null, .n = 0 };
                return ud;
            },
        }
    }

    pub fn create_smoothl1(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, user: ?*u8) !*anyopaque {
        switch (ctx.tipo) {
            .CPU => {
                var ud = try allocator.create(SL1Cpu.SmoothL1UserData);
                ud.* = .{ .pred = null, .target = null, .n = 0 };
                return ud;
            },
            .CPUSIMD => {
                var ud = try allocator.create(SL1simd.SmoothL1UserData);
                ud.* = .{ .pred = null, .target = null, .n = 0 };
                return ud;
            },
            else => {
                var ud = try allocator.create(SL1simd.SmoothL1UserData);
                ud.* = .{ .pred = null, .target = null, .n = 0 };
                return ud;
            },
        }
    }
};
