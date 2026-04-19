const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const computacao = @import("../../../computacao/mod.zig");

const CpuImplementacao = struct {
    data_bytes: []u8,
    count: usize,
};

fn cpu_get(impl_ptr: *anyopaque, i: usize) f64 {
    const s = @ptrCast(*CpuImplementacao, impl_ptr);
    const p = @ptrCast(*f64, s.data_bytes.ptr);
    return p[i];
}

fn cpu_set(impl_ptr: *anyopaque, i: usize, v: f64) void {
    const s = @ptrCast(*CpuImplementacao, impl_ptr);
    const p = @ptrCast(*f64, s.data_bytes.ptr);
    p[i] = v;
}

fn cpu_toArray(impl_ptr: *anyopaque, allocator: *std.mem.Allocator) anyerror![]f64 {
    const s = @ptrCast(*CpuImplementacao, impl_ptr);
    var out = try allocator.alloc(f64, s.count);
    const p = @ptrCast(*f64, s.data_bytes.ptr);
    for (p[0..s.count]) |val, idx| out[idx] = val;
    return out;
}

fn cpu_destroy(allocator: *std.mem.Allocator, impl_ptr: *anyopaque) void {
    const s = @ptrCast(*CpuImplementacao, impl_ptr);
    allocator.free(s.data_bytes);
    allocator.destroy(s);
}

const cpu_vtable = tensorImpl.TensorImplementacao{
    .get = cpu_get,
    .set = cpu_set,
    .toArray = cpu_toArray,
    .destroy = cpu_destroy,
};

pub const VTABLE = cpu_vtable;

pub fn create_impl(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, total: usize) !*anyopaque {
    _ = ctx;
    var impl = try allocator.create(CpuImplementacao);
    const bytes = try ctx.implementacao.*.allocate(allocator, total * @sizeOf(f64));
    impl.data_bytes = bytes;
    impl.count = total;
    const pz = @ptrCast(*f64, bytes.ptr);
    for (pz[0..total]) |*q| q.* = 0.0;
    return impl;
}

// Old constructors removed: use Tensor.init / Tensor.initFromArray in Tensor.zig

pub const error = error{ OutOfMemory, InvalidArgument };
