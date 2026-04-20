const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");

// Use shared BackendInstance from TensorImplementacao
const Backend = tensorImpl.BackendInstance;

pub fn cpu_get(impl_ptr: *Backend, i: usize) f64 {
    const s = impl_ptr.*;
    return s.data[i];
}

pub fn cpu_set(impl_ptr: *Backend, i: usize, v: f64) void {
    const s = impl_ptr.*;
    s.data[i] = v;
}

pub fn cpu_toArray(impl_ptr: *Backend, allocator: *std.mem.Allocator) anyerror![]f64 {
    const s = impl_ptr.*;
    var out = try allocator.alloc(f64, s.count);
    for (0..s.count) |i| out[i] = s.data[i];
    return out;
}

pub fn cpu_destroy(allocator: *std.mem.Allocator, impl_ptr: *Backend) void {
    const s = impl_ptr.*;
    allocator.free(s.data);
    allocator.free(s.grad);
    allocator.destroy(impl_ptr);
}

const cpu_vtable = tensorImpl.TensorImplementacao{
    .get = &cpu_get,
    .set = &cpu_set,
    .toArray = &cpu_toArray,
    .destroy = &cpu_destroy,
};

pub const VTABLE = cpu_vtable;

pub fn create_impl(allocator: *std.mem.Allocator, total: usize) !*Backend {
    var impl = try allocator.create(Backend);
    impl.data = try allocator.alloc(f64, total);
    impl.grad = try allocator.alloc(f64, total);
    impl.count = total;
    for (0..total) |i| {
        impl.data[i] = 0.0;
        impl.grad[i] = 0.0;
    }
    // attach vtable for dispatch
    impl.vtable = &VTABLE;
    return impl;
}

pub fn cpu_matmul_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").MatMulUserData, user.*);
    const a = ud.a.*;
    const b = ud.b.*;
    const m = ud.m;
    const n = ud.n;
    const p = ud.p;

    for (0..m * n) |i| a.grad[i] = 0.0;
    for (0..n * p) |i| b.grad[i] = 0.0;

    // a_grad
    for (i in 0..m) {
        for (k in 0..n) {
            var sum: f64 = 0.0;
            for (j in 0..p) {
                sum += grad[i * p + j] * b.data[k * p + j];
            }
            a.grad[i * n + k] += sum;
        }
    }

    // b_grad
    for (k in 0..n) {
        for (j in 0..p) {
            var sum: f64 = 0.0;
            for (i in 0..m) {
                sum += a.data[i * n + k] * grad[i * p + j];
            }
            b.grad[k * p + j] += sum;
        }
    }
}

pub fn cpu_conv_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").ConvUserData, user.*);
    const inb = ud.input.*;
    const kb = ud.kernel.*;
    const hin = ud.hin;
    const win = ud.win;
    const kh = ud.kh;
    const kw = ud.kw;
    const hout = hin - kh + 1;
    const wout = win - kw + 1;

    for (0..hin * win) |i| inb.grad[i] = 0.0;
    for (0..kh * kw) |i| kb.grad[i] = 0.0;

    for (0..kh) |ii| {
        for (0..kw) |jj| {
            var sum: f64 = 0.0;
            for (0..hout) |i| {
                for (0..wout) |j| {
                    const in_r = i + ii;
                    const in_c = j + jj;
                    sum += inb.data[in_r * win + in_c] * grad[i * wout + j];
                }
            }
            kb.grad[ii * kw + jj] += sum;
        }
    }

    for (0..hout) |i| {
        for (0..wout) |j| {
            const g = grad[i * wout + j];
            for (0..kh) |ii| {
                for (0..kw) |jj| {
                    const in_r = i + ii;
                    const in_c = j + jj;
                    inb.grad[in_r * win + in_c] += kb.data[ii * kw + jj] * g;
                }
            }
        }
    }
}

pub fn cpu_batchnorm_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").BatchNormUserData, user.*);
    const inb = ud.input.*;
    const outb = ud.out.*;
    const n = ud.n;
    const denom = ud.denom;

    var sum_dy: f64 = 0.0;
    var sum_dy_y: f64 = 0.0;
    for (0..n) |i| {
        const dy = grad[i];
        const y = outb.data[i];
        sum_dy += dy;
        sum_dy_y += dy * y;
    }

    const inv_n = 1.0 / @as(f64, n);
    const inv_denom = 1.0 / denom;

    for (0..n) |i| {
        const dy = grad[i];
        const y = outb.data[i];
        const dx = inv_denom * inv_n * ((@as(f64, n) * dy) - sum_dy - (y * sum_dy_y));
        inb.grad[i] += dx;
    }
}

pub fn cpu_bce_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").BCEUserData, user.*);
    const pred = ud.pred.*;
    const target = ud.target.*;
    const n = ud.n;

    const has_scalar_upstream = grad.len == 1;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        const eps = 1e-12;
        const pc = if (p < eps) eps else if (p > 1.0 - eps) 1.0 - eps else p;
        const dp = -(t / pc) + ((1.0 - t) / (1.0 - pc));
        pred.grad[i] += upstream * dp;
    }
}

pub fn cpu_smoothl1_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").SmoothL1UserData, user.*);
    const pred = ud.pred.*;
    const target = ud.target.*;
    const n = ud.n;

    const has_scalar_upstream = grad.len == 1;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const d = p - t;
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        var dp: f64 = 0.0;
        if (std.math.abs(d) < 1.0) {
            dp = d;
        } else {
            dp = if (d < 0.0) -1.0 else 1.0;
        }
        pred.grad[i] += upstream * dp;
    }
}

// Old constructors removed: use Tensor.init / Tensor.initFromArray in Tensor.zig

pub const TensorError = error{ OutOfMemory, InvalidArgument };
