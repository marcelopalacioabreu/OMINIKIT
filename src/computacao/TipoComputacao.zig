pub const TipoComputacao = enum {
    CPU,
    CPUSIMD,
    VULKAN,
    CUDA,
    ROCM,
    WGPU,
};

pub fn name(t: TipoComputacao) []const u8 {
    return switch (t) {
        .CPU => "CPU",
        .CPUSIMD => "CPUSIMD",
        .VULKAN => "VULKAN",
        .CUDA => "CUDA",
        .ROCM => "ROCM",
        .WGPU => "WGPU",
    };
}
