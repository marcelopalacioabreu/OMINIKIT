const std = @import("std");
const computacao = @import("computacao");

pub fn main() anyerror!void {
    var allocator_buf = std.heap.page_allocator;
    const allocator: *std.mem.Allocator = &allocator_buf;

    var ctx = computacao.ComputacaoContextoModule.ComputacaoCPUContexto();
    const size: usize = 512;
    const buf = try ctx.allocate(allocator, size);
    ctx.memset(buf, 0x7F);
    std.debug.print("Exemplo_0001: alocado {d} bytes no contexto {s}\n", .{ buf.len, @tagName(ctx.tipo) });
    ctx.free(allocator, buf);
}
