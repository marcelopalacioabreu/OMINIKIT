const std = @import("std");
const IFunc = @import("../Interfaces/IFuncaoRetropropagacao.zig").IFuncaoRetropropagacao;
const cpusimd = @import("../../nucleo/tensor/TensorCPUSIMD.zig");

pub const ConvFunction = struct {
    user: ?*u8,
    allocator: *std.mem.Allocator,

    pub fn init(allocator: *std.mem.Allocator, user: ?*u8) ConvFunction {
        return ConvFunction{ .user = user, .allocator = allocator };
    }

    pub fn Backward(self: *ConvFunction, grad: []const f64) void {
        cpusimd.simd_conv_backward(self.user, self.allocator, grad);
    }
};
// End of ConvFunction
