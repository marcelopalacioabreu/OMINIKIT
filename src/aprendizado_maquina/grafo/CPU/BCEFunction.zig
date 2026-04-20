const std = @import("std");
const cpu = @import("../../nucleo/tensor/TensorCPU.zig");
pub const BCEUserData = @import("../../nucleo/tensor/TensorImplementacao.zig").BCEUserData;

pub const BCEFunction = struct {
    user: ?*u8,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, user: ?*u8) BCEFunction {
        return BCEFunction{ .user = user, .allocator = allocator };
    }

    pub fn Backward(self: *BCEFunction, grad: []const f64) void {
        cpu.cpu_bce_backward(self.user, self.allocator, grad);
    }
};
