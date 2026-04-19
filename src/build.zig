const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});

    // module que aponta para o pacote OMINIKIT/computacao
    // NOTE: não criamos executáveis de exemplo aqui — exemplos são aplicações
    // independentes em `outros` que devem usar este pacote como dependência.
    const computacao_mod = b.createModule(.{
        .root_source_file = b.path("computacao/mod.zig"),
        .optimize = optimize,
        .target = b.graph.host,
    });

    // Não criar executáveis de exemplo neste build; o pacote está pronto para
    // ser importado por builds externos (ex.: exemplos em `outros`).
}
