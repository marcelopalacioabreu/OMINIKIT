const std = @import("std");
const computacao = @import("computacao");
const tensor = @import("tensor");

fn criar(c: *computacao.ComputacaoContextoModule.ComputacaoContexto, a: *std.mem.Allocator, shape: []const usize) !*tensor.Tensor {
    return tensor.Tensor.init(c, a, shape);
}

fn fromArray(c: *computacao.ComputacaoContextoModule.ComputacaoContexto, a: *std.mem.Allocator, shape: []const usize, data: []const f64) !*tensor.Tensor {
    return tensor.Tensor.fromArray(c, a, shape, data);
}

fn run_regression(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator) !void {

    // Create synthetic data y = 3 + 2*x + noise
    const n = 100;
    var xs = try allocator.alloc(f64, n);
    defer allocator.free(xs);
    var ys = try allocator.alloc(f64, n);
    defer allocator.free(ys);

    for (0..n) |i| {
        xs[i] = @as(f64, @floatFromInt(i)) / 10.0;
        // deterministic small noise-free data for portability
        ys[i] = 3.0 + 2.0 * xs[i];
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

    const X = try fromArray(ctx, allocator, &shape_X, Xdata);
    defer X.destroy(allocator);
    const y = try fromArray(ctx, allocator, &shape_y, ys);
    defer y.destroy(allocator);

    // Initialize weights w (2 x 1) to zeros
    const shape_w = [_]usize{ 2, 1 };
    var w = try criar(ctx, allocator, &shape_w);

    const lr = 0.0005;
    const iters = 5000;

    for (0..iters) |_| {
        const pred = try X.matMul(allocator, w);
        defer pred.destroy(allocator);

        const err = try pred.sub(allocator, y);
        defer err.destroy(allocator);

        const XT = try X.transpose(allocator);
        defer XT.destroy(allocator);

        const XT_err = try XT.matMul(allocator, err);
        defer XT_err.destroy(allocator);

        const grad = try XT_err.mulScalar(allocator, 2.0 / @as(f64, @floatFromInt(n)));
        defer grad.destroy(allocator);

        const lr_grad = try grad.mulScalar(allocator, lr);
        defer lr_grad.destroy(allocator);

        const new_w = try w.sub(allocator, lr_grad);
        w.destroy(allocator);
        w = new_w;
    }

    std.debug.print("Learned weights:\n", .{});
    const wArr = try w.toArray(allocator);
    defer allocator.free(wArr);
    std.debug.print("bias={d:0.4}, slope={d:0.4}\n", .{ wArr[0], wArr[1] });
    w.destroy(allocator);
}

pub fn main() anyerror!void {
    var allocator_buf = std.heap.page_allocator;
    const allocator: *std.mem.Allocator = &allocator_buf;

    std.debug.print("--- Running CPU experiment ---\n", .{});
    var ctx_cpu = computacao.ComputacaoContextoModule.ComputacaoCPUContexto();
    try run_regression(&ctx_cpu, allocator);

    std.debug.print("--- Running CPUSIMD experiment ---\n", .{});
    var ctx_simd = computacao.ComputacaoContextoModule.ComputacaoCPUSIMDContexto();
    try run_regression(&ctx_simd, allocator);
}
