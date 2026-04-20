const std = @import("std");
const cpu = @import("../../nucleo/tensor/TensorCPU.zig");
pub const SmoothL1UserData = @import("../../nucleo/tensor/TensorImplementacao.zig").SmoothL1UserData;

pub const SmoothL1Function = struct {
    user: ?*u8,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, user: ?*u8) SmoothL1Function {
        return SmoothL1Function{ .user = user, .allocator = allocator };
    }

    pub fn Backward(self: *SmoothL1Function, grad: []const f64) void {
        cpu.cpu_smoothl1_backward(self.user, self.allocator, grad);
    }
};
