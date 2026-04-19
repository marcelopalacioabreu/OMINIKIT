const std = @import("std");

pub const TensorImplementacao = extern struct {
    get: fn(impl_ptr: *anyopaque, i: usize) f64,
    set: fn(impl_ptr: *anyopaque, i: usize, v: f64) void,
    toArray: fn(impl_ptr: *anyopaque, allocator: *std.mem.Allocator) anyerror![]f64,
    destroy: fn(allocator: *std.mem.Allocator, impl_ptr: *anyopaque) void,
};


pub const error = error{ OutOfMemory };
