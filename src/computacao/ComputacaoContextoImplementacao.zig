const std = @import("std");

pub const ComputacaoContextoImplementacao = struct {
    allocate: *const fn (allocator: *std.mem.Allocator, size: usize) anyerror![]u8,
    free: *const fn (allocator: *std.mem.Allocator, buffer: []u8) void,
    memset: *const fn (buffer: []u8, value: u8) void,
    memcpy: *const fn (dst: []u8, src: []const u8) void,
};
