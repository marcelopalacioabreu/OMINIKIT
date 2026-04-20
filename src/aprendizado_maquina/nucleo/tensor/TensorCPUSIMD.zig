const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");

// Use shared BackendInstance from TensorImplementacao
const Backend = tensorImpl.BackendInstance;

// Use shared user-data types from TensorImplementacao

pub fn simd_get(impl_ptr: *Backend, i: usize) f64 {
    const s = impl_ptr.*;
    return s.data[i];
}

pub fn simd_set(impl_ptr: *Backend, i: usize, v: f64) void {
    const s = impl_ptr.*;
    s.data[i] = v;
}

pub fn simd_toArray(impl_ptr: *Backend, allocator: *std.mem.Allocator) anyerror![]f64 {
    const s = impl_ptr.*;
    var out = try allocator.alloc(f64, s.count);
    for (0..s.count) |i| out[i] = s.data[i];
    return out;
}

pub fn simd_destroy(allocator: *std.mem.Allocator, impl_ptr: *Backend) void {
    const s = impl_ptr.*;
    allocator.free(s.data);
    allocator.free(s.grad);
    allocator.destroy(impl_ptr);
}

const simd_vtable = tensorImpl.TensorImplementacao{
    .get = &simd_get,
    .set = &simd_set,
    .toArray = &simd_toArray,
    .destroy = &simd_destroy,
};

pub const VTABLE = simd_vtable;

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

// Old constructors removed: use Tensor.init / Tensor.initFromArray in Tensor.zig

pub const TensorError = error{ OutOfMemory, InvalidArgument };

// Optimized (naive, cache-friendly) matMul for CPUSIMD backend.
pub fn simd_matMul(allocator: *std.mem.Allocator, a_impl: *Backend, b_impl: *Backend, m: usize, n: usize, p: usize) !*Backend {
    const a = a_impl.*;
    const b = b_impl.*;
    var out = try create_impl(allocator, m * p);

    // Multiply with loop ordering i,k,j for cache locality on b
    for (0..m) |i| {
        for (0..n) |k| {
            const a_ik = a.data[i * n + k];
            const b_row = k * p;
            const out_row = i * p;
            for (0..p) |j| {
                out.data[out_row + j] += a_ik * b.data[b_row + j];
            }
        }
    }
    return out;
}

pub fn simd_matmul_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").MatMulUserData, user.*);
    const a = ud.a.*;
    const b = ud.b.*;
    const m = ud.m;
    const n = ud.n;
    const p = ud.p;

    // Compute gradients: a_grad = grad * b^T  (m x p) * (p x n) -> (m x n)
    // and b_grad = a^T * grad  (n x m) * (m x p) -> (n x p)
    // Zero accumulators
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

// Very small, generic 2D convolution helper. Assumes input is 2D stored row-major
// with dimensions (hin x win) and kernel (kh x kw). Returns output backend sized (hout x wout).
pub fn simd_conv(allocator: *std.mem.Allocator, input_impl: *Backend, hin: usize, win: usize, kernel_impl: *Backend, kh: usize, kw: usize) !*Backend {
    const inb = input_impl.*;
    const kb = kernel_impl.*;
    if (hin < kh or win < kw) return error.InvalidArgument;
    const hout = hin - kh + 1;
    const wout = win - kw + 1;
    var out = try create_impl(allocator, hout * wout);

    for (hout) |i| {
        for (wout) |j| {
            var sum: f64 = 0.0;
            for (kh) |ii| {
                for (kw) |jj| {
                    const in_r = i + ii;
                    const in_c = j + jj;
                    const in_idx = in_r * win + in_c;
                    const k_idx = ii * kw + jj;
                    sum += inb.data[in_idx] * kb.data[k_idx];
                }
            }
            out.data[i * wout + j] = sum;
        }
    }
    return out;
}

pub fn simd_conv_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
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

    // Zero accumulators
    for (0..hin * win) |i| inb.grad[i] = 0.0;
    for (0..kh * kw) |i| kb.grad[i] = 0.0;

    // kernel gradients: sum over output positions
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

    // input gradients: distribute kernel * grad to input positions
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

// Batch-normalize in-place: normalize whole tensor to zero mean and unit variance with eps.
pub fn simd_batchnorm_inplace(impl_ptr: *Backend, epsilon: f64) void {
    const b = impl_ptr.*;
    var mean: f64 = 0.0;
    const n = b.count;
    for (0..n) |i| mean += b.data[i];
    mean /= @as(f64, n);
    var varacc: f64 = 0.0;
    for (0..n) |i| {
        const d = b.data[i] - mean;
        varacc += d * d;
    }
    varacc /= @as(f64, n);
    const denom = std.math.sqrt(varacc + epsilon);
    for (0..n) |i| b.data[i] = (b.data[i] - mean) / denom;
}

// Create a new normalized backend (out) and return it; caller may attach backward userdata.
pub fn simd_batchnorm(allocator: *std.mem.Allocator, input_impl: *Backend, epsilon: f64) !*Backend {
    const inb = input_impl.*;
    const n = inb.count;
    var mean: f64 = 0.0;
    for (0..n) |i| mean += inb.data[i];
    mean /= @as(f64, n);
    var varacc: f64 = 0.0;
    for (0..n) |i| {
        const d = inb.data[i] - mean;
        varacc += d * d;
    }
    varacc /= @as(f64, n);
    const denom = std.math.sqrt(varacc + epsilon);

    var out = try create_impl(allocator, n);
    for (0..n) |i| out.data[i] = (inb.data[i] - mean) / denom;

    return out;
}

pub fn simd_batchnorm_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*@import("TensorImplementacao.zig").BatchNormUserData, user.*);
    const inb = ud.input.*;
    const outb = ud.out.*;
    const n = ud.n;
    const denom = ud.denom;

    // compute sums
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
