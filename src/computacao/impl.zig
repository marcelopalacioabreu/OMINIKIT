const std = @import("std");

pub const Impl = struct {
    allocate: fn (allocator: *std.mem.Allocator, size: usize) anyerror![]u8,
    free: fn (allocator: *std.mem.Allocator, buffer: []u8) void,
    memset: fn (buffer: []u8, value: u8) void,
    memcpy: fn (dst: []u8, src: []const u8) void,
};
