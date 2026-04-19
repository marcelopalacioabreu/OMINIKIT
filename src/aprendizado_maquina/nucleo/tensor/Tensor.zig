const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const cpu = @import("TensorCPU.zig");
const cpusimd = @import("TensorCPUSIMD.zig");
const computacao = @import("../../../computacao/mod.zig");

pub const Tensor = struct {
    vtable: *const tensorImpl.TensorImplementacao,
    impl_ptr: *anyopaque,
    shape: []usize,
    size: usize,
    requires_grad: bool,

    pub fn init(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize) !*Tensor {
        var total: usize = 1;
        for (shape_in) |d| total *= if (d == 0) 1 else d;

        // allocate shape storage
        var shape_buf = try allocator.alloc(usize, shape_in.len);
        for (shape_buf) |*s, i| s.* = shape_in[i];

        // create implementation via backend
        var impl_ptr: *anyopaque = null;
        var vtable: *const tensorImpl.TensorImplementacao = null;
        switch (ctx.tipo) {
            .CPU => {
                impl_ptr = try cpu.create_impl(ctx, allocator, total);
                vtable = &cpu.VTABLE;
            }
            .CPUSIMD => {
                impl_ptr = try cpusimd.create_impl(ctx, allocator, total);
                vtable = &cpusimd.VTABLE;
            }
            else => {
                impl_ptr = try cpu.create_impl(ctx, allocator, total);
                vtable = &cpu.VTABLE;
            }
        }

        var obj = try allocator.create(Tensor);
        obj.vtable = vtable;
        obj.impl_ptr = impl_ptr;
        obj.shape = shape_buf[0..shape_in.len];
        obj.size = total;
        obj.requires_grad = false;
        return obj;
    }

    pub fn fromArray(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize, data: []const f64) !*Tensor {
        var obj = try Tensor.init(ctx, allocator, shape_in);
        if (data.len != obj.size) return error.InvalidArgument;
        for (data) |v, i| obj.vtable.set(obj.impl_ptr, i, v);
        return obj;
    }

    pub fn get(self: *Tensor, i: usize) f64 {
        return self.vtable.get(self.impl_ptr, i);
    }

    pub fn set(self: *Tensor, i: usize, v: f64) void {
        self.vtable.set(self.impl_ptr, i, v);
    }

    pub fn toArray(self: *Tensor, allocator: *std.mem.Allocator) anyerror![]f64 {
        return self.vtable.toArray(self.impl_ptr, allocator);
    }

    pub fn destroy(self: *Tensor, allocator: *std.mem.Allocator) void {
        self.vtable.destroy(allocator, self.impl_ptr);
        allocator.free(self.shape);
        allocator.destroy(self);
    }
};
