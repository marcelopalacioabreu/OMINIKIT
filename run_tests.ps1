#!/usr/bin/env pwsh
Set-StrictMode -Version Latest

Write-Host "Running Zig tests for OMINIKIT (CPU and CPUSIMD)." -ForegroundColor Cyan

# Run the test file directly with zig test (does not require build.zig)
zig test tests\test_layers.zig
