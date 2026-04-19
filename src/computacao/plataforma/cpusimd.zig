const std = @import("std");
const impl = @import("../impl.zig");
const cpu = @import("cpu.zig");
pub const Allocator = std.mem.Allocator;

pub fn allocate(allocator: *Allocator, size: usize) ![]u8 {
    return cpu.allocate(allocator, size);
}

pub fn free(allocator: *Allocator, buffer: []u8) void {
    cpu.free(allocator, buffer);
}

pub fn memset(buffer: []u8, value: u8) void {
    cpu.memset(buffer, value);
}

pub fn memcpy(dst: []u8, src: []const u8) void {
    cpu.memcpy(dst, src);
}

pub const IMPL = impl.Impl{
    .allocate = allocate,
    .free = free,
    .memset = memset,
    .memcpy = memcpy,
};
