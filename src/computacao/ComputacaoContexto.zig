const std = @import("std");
const tipo_mod = @import("TipoComputacao.zig");
pub const TipoComputacao = tipo_mod.TipoComputacao;

const impl = @import("impl.zig");
const cpu = @import("plataforma/cpu.zig");
const cpusimd = @import("plataforma/cpusimd.zig");
const vulkan = @import("plataforma/vulkan.zig");

pub const ComputacaoContexto = struct {
    tipo: TipoComputacao,

    pub fn init(tipo: TipoComputacao) ComputacaoContexto {
        return ComputacaoContexto{ .tipo = tipo };
    }

    pub fn inicializar(tipo: TipoComputacao) ComputacaoContexto {
        return ComputacaoContexto.init(tipo);
    }

    pub fn allocate(self: *ComputacaoContexto, allocator: *std.mem.Allocator, size: usize) anyerror![]u8 {
        return switch (self.tipo) {
            .CPU => cpu.allocate(allocator, size),
            .CPUSIMD => cpusimd.allocate(allocator, size),
            else => vulkan.allocate(allocator, size),
        };
    }

    pub fn free(self: *ComputacaoContexto, allocator: *std.mem.Allocator, buffer: []u8) void {
        switch (self.tipo) {
            .CPU => cpu.free(allocator, buffer),
            .CPUSIMD => cpusimd.free(allocator, buffer),
            else => vulkan.free(allocator, buffer),
        }
    }

    pub fn memset(self: *ComputacaoContexto, buffer: []u8, value: u8) void {
        switch (self.tipo) {
            .CPU => cpu.memset(buffer, value),
            .CPUSIMD => cpusimd.memset(buffer, value),
            else => vulkan.memset(buffer, value),
        }
    }

    pub fn memcpy(self: *ComputacaoContexto, dst: []u8, src: []const u8) void {
        switch (self.tipo) {
            .CPU => cpu.memcpy(dst, src),
            .CPUSIMD => cpusimd.memcpy(dst, src),
            else => vulkan.memcpy(dst, src),
        }
    }
};

pub fn ComputacaoCPUContexto() ComputacaoContexto {
    return ComputacaoContexto.inicializar(.CPU);
}

pub fn ComputacaoCPUSIMDContexto() ComputacaoContexto {
    return ComputacaoContexto.inicializar(.CPUSIMD);
}
