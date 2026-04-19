const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const cpu = @import("TensorCPU.zig");
const cpusimd = @import("TensorCPUSIMD.zig");
const computacao = @import("../../../computacao/ComputacaoContexto.zig");
const tipo_mod = @import("../../../computacao/TipoComputacao.zig");

pub const Tensor = struct {
    tipo: tipo_mod.TipoComputacao,
    impl_ptr: *tensorImpl.BackendInstance,
    shape: []usize,
    size: usize,
    requires_grad: bool,

    pub fn init(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize) !*Tensor {
        var total: usize = 1;
        for (shape_in) |d| total *= if (d == 0) 1 else d;

        // allocate shape storage
        var shape_buf = try allocator.alloc(usize, shape_in.len);
        for (0..shape_in.len) |i| shape_buf[i] = shape_in[i];

        // create implementation via backend
        var impl_ptr: *tensorImpl.BackendInstance = undefined;
        var vtype: tipo_mod.TipoComputacao = undefined;
        switch (ctx.tipo) {
            .CPU => {
                impl_ptr = try cpu.create_impl(allocator, total);
                vtype = .CPU;
            },
            .CPUSIMD => {
                impl_ptr = try cpusimd.create_impl(allocator, total);
                vtype = .CPUSIMD;
            },
            else => {
                impl_ptr = try cpu.create_impl(allocator, total);
                vtype = .CPU;
            },
        }

        var obj = try allocator.create(Tensor);
        obj.tipo = vtype;
        obj.impl_ptr = impl_ptr;
        obj.shape = shape_buf[0..shape_in.len];
        obj.size = total;
        obj.requires_grad = false;
        return obj;
    }

    pub fn fromArray(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize, data: []const f64) !*Tensor {
        const obj = try Tensor.init(ctx, allocator, shape_in);
        if (data.len != obj.size) return error.InvalidArgument;
        for (0..data.len) |i| {
            switch (obj.tipo) {
                .CPU => cpu.cpu_set(obj.impl_ptr, i, data[i]),
                .CPUSIMD => cpusimd.simd_set(obj.impl_ptr, i, data[i]),
                else => cpu.cpu_set(obj.impl_ptr, i, data[i]),
            }
        }
        return obj;
    }

    pub fn get(self: *Tensor, i: usize) f64 {
        return switch (self.tipo) {
            .CPU => cpu.cpu_get(self.impl_ptr, i),
            .CPUSIMD => cpusimd.simd_get(self.impl_ptr, i),
            else => cpu.cpu_get(self.impl_ptr, i),
        };
    }

    pub fn set(self: *Tensor, i: usize, v: f64) void {
        switch (self.tipo) {
            .CPU => cpu.cpu_set(self.impl_ptr, i, v),
            .CPUSIMD => cpusimd.simd_set(self.impl_ptr, i, v),
            else => cpu.cpu_set(self.impl_ptr, i, v),
        }
    }

    pub fn toArray(self: *Tensor, allocator: *std.mem.Allocator) anyerror![]f64 {
        return switch (self.tipo) {
            .CPU => cpu.cpu_toArray(self.impl_ptr, allocator),
            .CPUSIMD => cpusimd.simd_toArray(self.impl_ptr, allocator),
            else => cpu.cpu_toArray(self.impl_ptr, allocator),
        };
    }

    pub fn destroy(self: *Tensor, allocator: *std.mem.Allocator) void {
        switch (self.tipo) {
            .CPU => cpu.cpu_destroy(allocator, self.impl_ptr),
            .CPUSIMD => cpusimd.simd_destroy(allocator, self.impl_ptr),
            else => cpu.cpu_destroy(allocator, self.impl_ptr),
        }
        allocator.free(self.shape);
        allocator.destroy(self);
    }

    pub fn add(self: *Tensor, allocator: *std.mem.Allocator, other: *Tensor) !*Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return error.IncompatibleShapes;
        const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
        for (0..self.size) |i| {
            res.set(i, self.get(i) + other.get(i));
        }
        return res;
    }

    pub fn sub(self: *Tensor, allocator: *std.mem.Allocator, other: *Tensor) !*Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return error.IncompatibleShapes;
        const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
        for (0..self.size) |i| {
            res.set(i, self.get(i) - other.get(i));
        }
        return res;
    }

    pub fn mulScalar(self: *Tensor, allocator: *std.mem.Allocator, scalar: f64) !*Tensor {
        const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
        for (0..self.size) |i| {
            res.set(i, self.get(i) * scalar);
        }
        return res;
    }

    pub fn matMul(self: *Tensor, allocator: *std.mem.Allocator, other: *Tensor) !*Tensor {
        if (self.shape.len != 2 or other.shape.len != 2) return error.Not2D;
        const m = self.shape[0];
        const n = self.shape[1];
        const n2 = other.shape[0];
        const p = other.shape[1];
        if (n != n2) return error.IncompatibleInnerDims;

        const res_shape = [_]usize{ m, p };
        const res = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
        for (0..m) |i| {
            for (0..p) |j| {
                var sum: f64 = 0.0;
                for (0..n) |k| {
                    sum += self.get(i * n + k) * other.get(k * p + j);
                }
                res.set(i * p + j, sum);
            }
        }
        return res;
    }

    pub fn transpose(self: *Tensor, allocator: *std.mem.Allocator) !*Tensor {
        if (self.shape.len != 2) return error.Not2D;
        const m = self.shape[0];
        const n = self.shape[1];
        const res_shape = [_]usize{ n, m };
        const res = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
        for (0..m) |i| {
            for (0..n) |j| {
                res.set(j * m + i, self.get(i * n + j));
            }
        }
        return res;
    }

    fn init_with_type(tipo: tipo_mod.TipoComputacao, allocator: *std.mem.Allocator, shape_in: []const usize) !*Tensor {
        var total: usize = 1;
        for (shape_in) |d| total *= if (d == 0) 1 else d;
        var shape_buf = try allocator.alloc(usize, shape_in.len);
        for (0..shape_in.len) |i| shape_buf[i] = shape_in[i];

        var impl_ptr: *tensorImpl.BackendInstance = undefined;
        switch (tipo) {
            .CPU => impl_ptr = try cpu.create_impl(allocator, total),
            .CPUSIMD => impl_ptr = try cpusimd.create_impl(allocator, total),
            else => impl_ptr = try cpu.create_impl(allocator, total),
        }

        var obj = try allocator.create(Tensor);
        obj.tipo = tipo;
        obj.impl_ptr = impl_ptr;
        obj.shape = shape_buf;
        obj.size = total;
        obj.requires_grad = false;
        return obj;
    }
};
