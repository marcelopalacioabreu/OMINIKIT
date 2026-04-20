const std = @import("std");
const IFunc = @import("../Interfaces/IFuncaoRetropropagacao.zig").IFuncaoRetropropagacao;
const cpu = @import("../../nucleo/tensor/TensorCPU.zig");

pub const BNFunction = struct {
    user: ?*u8,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, user: ?*u8) BNFunction {
        return BNFunction{ .user = user, .allocator = allocator };
    }

    pub fn Backward(self: *BNFunction, grad: []const f64) void {
        cpu.cpu_batchnorm_backward(self.user, self.allocator, grad);
    }
};
