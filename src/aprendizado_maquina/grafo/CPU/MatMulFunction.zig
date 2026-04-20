const std = @import("std");
const IFunc = @import("../Interfaces/IFuncaoRetropropagacao.zig").IFuncaoRetropropagacao;
const cpu = @import("../../nucleo/tensor/TensorCPU.zig");

pub const MatMulFunction = struct {
    user: ?*u8,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, user: ?*u8) MatMulFunction {
        return MatMulFunction{ .user = user, .allocator = allocator };
    }

    pub fn Backward(self: *MatMulFunction, grad: []const f64) void {
        cpu.cpu_matmul_backward(self.user, self.allocator, grad);
    }
};
