const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const conv_module = b.createModule(.{ .root_source_file = b.path("test_conv_root.zig"), .target = target });
    const exe_conv = b.addExecutable(.{ .name = "test_conv", .root_module = conv_module });
    b.installArtifact(exe_conv);

    const loss_module = b.createModule(.{ .root_source_file = b.path("test_losses_root.zig"), .target = target });
    const exe_loss = b.addExecutable(.{ .name = "test_losses", .root_module = loss_module });
    b.installArtifact(exe_loss);
}
