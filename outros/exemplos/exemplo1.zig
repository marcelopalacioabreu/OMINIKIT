const std = @import("std");
const computacao = @import("../../src/computacao/ComputacaoContexto.zig");
const tensor = @import("../../src/aprendizado_maquina/nucleo/tensor/Tensor.zig");
const fabrica_mod = @import("../../src/aprendizado_maquina/nucleo/tensor/FabricaTensor.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var allocator = gpa.allocator();

    var ctx = computacao.ComputacaoCPUContexto();
    var fabrica = fabrica_mod.FabricaTensor.init(&ctx, &allocator);

    // Create synthetic data y = 3 + 2*x + noise
    const n = 100;
    var xs = try allocator.alloc(f64, n);
    defer allocator.free(xs);
    var ys = try allocator.alloc(f64, n);
    defer allocator.free(ys);

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    for (0..n) |i| {
        xs[i] = @as(f64, @floatFromInt(i)) / 10.0;
        ys[i] = 3.0 + 2.0 * xs[i] + (rand.float(f64) - 0.5) * 1.0;
    }

    // Build design matrix X (n x 2), column0 = 1 for bias, column1 = x
    var Xdata = try allocator.alloc(f64, n * 2);
    defer allocator.free(Xdata);
    for (0..n) |i| {
        Xdata[i * 2 + 0] = 1.0;
        Xdata[i * 2 + 1] = xs[i];
    }

    const shape_X = [_]usize{ n, 2 };
    const shape_y = [_]usize{ n, 1 };

    const X = try fabrica.fromArray(&shape_X, Xdata);
    defer X.destroy(&allocator);
    const y = try fabrica.fromArray(&shape_y, ys);
    defer y.destroy(&allocator);

    // Initialize weights w (2 x 1) to zeros
    const shape_w = [_]usize{ 2, 1 };
    var w = try fabrica.criar(&shape_w);
    // defer w.destroy(&allocator); // we reassign w in the loop

    const lr = 0.0005;
    const iters = 5000;

    for (0..iters) |_| {
        // pred = X * w
        const pred = try X.matMul(&allocator, w);
        defer pred.destroy(&allocator);

        // error = pred - y
        const err = try pred.sub(&allocator, y);
        defer err.destroy(&allocator);

        // grad = (2/n) * X^T * error
        const XT = try X.transpose(&allocator);
        defer XT.destroy(&allocator);

        const XT_err = try XT.matMul(&allocator, err);
        defer XT_err.destroy(&allocator);

        const grad = try XT_err.mulScalar(&allocator, 2.0 / @as(f64, @floatFromInt(n)));
        defer grad.destroy(&allocator);

        // update w = w - lr * grad
        const lr_grad = try grad.mulScalar(&allocator, lr);
        defer lr_grad.destroy(&allocator);

        const new_w = try w.sub(&allocator, lr_grad);
        w.destroy(&allocator);
        w = new_w;
    }

    std.debug.print("Learned weights:\n", .{});
    const wArr = try w.toArray(&allocator);
    defer allocator.free(wArr);
    std.debug.print("bias={d:0.4}, slope={d:0.4}\n", .{ wArr[0], wArr[1] });
    w.destroy(&allocator);
}
