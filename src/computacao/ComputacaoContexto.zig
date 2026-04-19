const std = @import("std");
const tipo_mod = @import("TipoComputacao.zig");
pub const TipoComputacao = tipo_mod.TipoComputacao;

const cpu = @import("plataforma/cpu.zig");
const cpusimd = @import("plataforma/cpusimd.zig");
const vulkan = @import("plataforma/vulkan.zig");
const impl_mod = @import("ComputacaoContextoImplementacao.zig");

pub const ComputacaoContexto = struct {
    tipo: TipoComputacao,
    implementacao: *const impl_mod.ComputacaoContextoImplementacao,

    pub fn init(tipo: TipoComputacao) ComputacaoContexto {
        const impl_ptr: *const impl_mod.ComputacaoContextoImplementacao = switch (tipo) {
            .CPU => &cpu.COMPUTACAO_CONTEXTO_IMPLEMENTACAO,
            .CPUSIMD => &cpusimd.COMPUTACAO_CONTEXTO_IMPLEMENTACAO,
            else => &vulkan.COMPUTACAO_CONTEXTO_IMPLEMENTACAO,
        };
        return ComputacaoContexto{ .tipo = tipo, .implementacao = impl_ptr };
    }

    pub fn allocate(self: *ComputacaoContexto, allocator: *std.mem.Allocator, size: usize) anyerror![]u8 {
        return self.implementacao.*.allocate(allocator, size);
    }

    pub fn free(self: *ComputacaoContexto, allocator: *std.mem.Allocator, buffer: []u8) void {
        self.implementacao.*.free(allocator, buffer);
    }

    pub fn memset(self: *ComputacaoContexto, buffer: []u8, value: u8) void {
        self.implementacao.*.memset(buffer, value);
    }

    pub fn memcpy(self: *ComputacaoContexto, dst: []u8, src: []const u8) void {
        self.implementacao.*.memcpy(dst, src);
    }
};

pub fn ComputacaoCPUContexto() ComputacaoContexto {
    return ComputacaoContexto.init(.CPU);
}

pub fn ComputacaoCPUSIMDContexto() ComputacaoContexto {
    return ComputacaoContexto.init(.CPUSIMD);
}
