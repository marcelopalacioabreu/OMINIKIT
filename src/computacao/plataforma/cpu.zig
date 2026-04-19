const std = @import("std");
const impl = @import("../impl.zig");
pub const Allocator = std.mem.Allocator;

pub fn allocate(allocator: *Allocator, size: usize) ![]u8 {
    return try allocator.alloc(u8, size);
}

pub fn free(allocator: *Allocator, buffer: []u8) void {
    allocator.free(buffer);
}

pub fn memset(buffer: []u8, value: u8) void {
    for (buffer) |*b| b.* = value;
}

pub fn memcpy(dst: []u8, src: []const u8) void {
    std.mem.copy(u8, dst, src);
}

pub const IMPL = impl.Impl{
    .allocate = allocate,
    .free = free,
    .memset = memset,
    .memcpy = memcpy,
};
