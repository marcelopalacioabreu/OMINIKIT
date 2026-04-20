const std = @import("std");

pub fn build(b: *std.Build) void {
    // Create temporary root shims that re-export the tests in `testes/`.
    const wf_conv = b.addWriteFile("test_conv_root.zig", "const _ = @import(\"testes/test_conv_root.zig\");\n");
    const wf_losses = b.addWriteFile("test_losses_root.zig", "const _ = @import(\"testes/test_losses_root.zig\");\n");
    const wf_layers = b.addWriteFile("test_layers_root.zig", "const _ = @import(\"testes/test_layers_root.zig\");\n");

    const run_conv = b.addSystemCommand(&[_][]const u8{ "zig", "test", "test_conv_root.zig" });
    const run_losses = b.addSystemCommand(&[_][]const u8{ "zig", "test", "test_losses_root.zig" });
    const run_layers = b.addSystemCommand(&[_][]const u8{ "zig", "test", "test_layers_root.zig" });

    // Ensure system commands run after the files are written
    run_conv.step.dependOn(&wf_conv.step);
    run_losses.step.dependOn(&wf_losses.step);
    run_layers.step.dependOn(&wf_layers.step);

    b.getInstallStep().dependOn(&run_conv.step);
    b.getInstallStep().dependOn(&run_losses.step);
    b.getInstallStep().dependOn(&run_layers.step);
}
