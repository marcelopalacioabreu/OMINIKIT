const std = @import("std");

const computacao = @import("../src/computacao/ComputacaoContexto.zig");
const FabricaTensor = @import("../src/aprendizado_maquina/nucleo/tensor/FabricaTensor.zig").FabricaTensor;
const Tensor = @import("../src/aprendizado_maquina/nucleo/tensor/Tensor.zig").Tensor;

test "matmul and backward CPU" {
    var allocator = std.heap.page_allocator;
    var ctx = computacao.ComputacaoCPUContexto();
    var ft = FabricaTensor.init(&ctx, &allocator);

    const a = try ft.criar(&[_]usize{ 2, 3 });
    defer a.destroy(&allocator);
    const b = try ft.criar(&[_]usize{ 3, 2 });
    defer b.destroy(&allocator);

    // fill a
    for (0..a.size) |i| a.set(i, @as(f64, i + 1));
    // fill b
    for (0..b.size) |i| b.set(i, 1.0);

    const c = try a.matMul(&allocator, b);
    defer c.destroy(&allocator);

    // call backward with scalar upstream
    const grad = [_]f64{1.0};
    c.backward(&allocator, &grad);

    // basic sanity: grads on a and b arrays should be non-zero for some elements
    var any_nonzero: bool = false;
    for (0..a.size) |i| if (a.impl_ptr.grad[i] != 0.0) any_nonzero = true;
    try std.testing.expect(any_nonzero);
}

test "matmul and backward CPUSIMD" {
    var allocator = std.heap.page_allocator;
    var ctx = computacao.ComputacaoCPUSIMDContexto();
    var ft = FabricaTensor.init(&ctx, &allocator);

    const a = try ft.criar(&[_]usize{ 2, 3 });
    defer a.destroy(&allocator);
    const b = try ft.criar(&[_]usize{ 3, 2 });
    defer b.destroy(&allocator);

    for (0..a.size) |i| a.set(i, @as(f64, i + 1));
    for (0..b.size) |i| b.set(i, 1.0);

    const c = try a.matMul(&allocator, b);
    defer c.destroy(&allocator);

    const grad = [_]f64{1.0};
    c.backward(&allocator, &grad);

    var any_nonzero: bool = false;
    for (0..a.size) |i| if (a.impl_ptr.grad[i] != 0.0) any_nonzero = true;
    try std.testing.expect(any_nonzero);
}
