const std = @import("std");

// Adapter that uses vendored zignal (vendor/zignal) to load images.
// If zignal is not present, this function will fall back to the older
// ManipuladorDeImagem implementation.

const Manipulador = @import("./ManipuladorDeImagem.zig");

pub fn loadAsGray(allocator: *std.mem.Allocator, path: []const u8) !Manipulador.GrayImage {
    // Try to import vendored zignal. If not found, fall back.
    const zimg = @import("../../../../vendor/zignal/src/image.zig");
    const zcolor = @import("../../../../vendor/zignal/src/color.zig");
    const Rgba = zcolor.Rgba(u8);
    const Img = zimg.Image(Rgba);

    // Read file into memory using current working directory and File.readToEndAlloc
    const io = std.Io; // use std.Io namespace
    const cwd = io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{}) catch {
        return Manipulador.carregarComoGray(allocator, path);
    };
    defer file.close(io);

    const data = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(data);

    // Load image via zignal
    var img = try Img.loadFromBytes(allocator, data) catch {
        // fallback to prior loader on any zignal error
        return Manipulador.carregarComoGray(allocator, path);
    };
    defer img.deinit(allocator);

    const rows = @as(usize, img.rows);
    const cols = @as(usize, img.cols);
    const px_count = rows * cols;

    var out = try allocator.alloc(u8, px_count);

    // Convert RGB(A) -> grayscale using Rec. 709 luma coefficients
    for (px_count) |i| {
        const p = img.data[i];
        const r = @as(f32, p.r);
        const g = @as(f32, p.g);
        const b = @as(f32, p.b);
        const grayf = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        var gval: u8 = @as(u8, std.math.clamp(@as(f32, 0.0), @as(f32, 255.0), grayf));
        // simple rounding
        if (grayf >= @as(f32, gval) + 0.5 and gval < 255) gval += 1;
        out[i] = gval;
    }

    return Manipulador.GrayImage{ .buf = out, .width = cols, .height = rows };
}
