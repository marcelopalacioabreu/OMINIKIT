const std = @import("std");
const implementacao = @import("../ComputacaoContextoImplementacao.zig");
pub const Allocator = std.mem.Allocator;

pub fn allocate(allocator: *Allocator, size: usize) anyerror![]u8 {
    return try allocator.alloc(u8, size);
}

pub fn free(allocator: *Allocator, buffer: []u8) void {
    allocator.free(buffer);
}

pub fn memset(buffer: []u8, value: u8) void {
    for (buffer) |*b| b.* = value;
}

pub fn memcpy(dst: []u8, src: []const u8) void {
    for (0..src.len) |i| dst[i] = src[i];
}

pub var COMPUTACAO_CONTEXTO_IMPLEMENTACAO = implementacao.ComputacaoContextoImplementacao{
    .allocate = &allocate,
    .free = &free,
    .memset = &memset,
    .memcpy = &memcpy,
};
