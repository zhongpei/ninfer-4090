#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr int kScales = 127; // Public cache domain: nonnegative, finite E4M3 codes 0..0x7e.
constexpr int kCases = 256 * kScales;

__global__ void decode(const std::uint8_t* codes, std::uint16_t* out8,
                       std::uint16_t* out16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kCases) { return; }
    const auto scale = static_cast<std::uint8_t>(i / 256);
    const auto* source = codes + i * 8;
    const int4 first = ninfer::ops::kv_cache_nvfp4_dequant_f16x8(source, scale);
    const int4 second = ninfer::ops::kv_cache_nvfp4_dequant_f16x8(source + 4, scale);
    const auto full = ninfer::ops::kv_cache_nvfp4_dequant_f16x16(source, scale);
    reinterpret_cast<int4*>(out8)[2 * i] = first;
    reinterpret_cast<int4*>(out8)[2 * i + 1] = second;
    reinterpret_cast<int4*>(out16)[2 * i] = full.lo;
    reinterpret_cast<int4*>(out16)[2 * i + 1] = full.hi;
}

// Independent exact mathematical oracle; no CUDA FP4/FP8 conversion or production helpers.
std::uint16_t oracle(unsigned nibble, unsigned scale_code) {
    constexpr double magnitudes[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const unsigned exponent = scale_code >> 3;
    const unsigned fraction = scale_code & 7;
    const double scale = exponent == 0 ? std::ldexp(double(fraction), -9)
                                      : std::ldexp(1.0 + fraction / 8.0, int(exponent) - 7);
    const double value = magnitudes[nibble & 7] * scale;
    const unsigned sign = (nibble & 8) ? 0x8000 : 0;
    if (value < std::ldexp(1.0, -14)) {
        return static_cast<std::uint16_t>(sign | unsigned(std::llround(std::ldexp(value, 24))));
    }
    int binary_exponent;
    const double mantissa = std::frexp(value, &binary_exponent) * 2;
    return static_cast<std::uint16_t>(sign | unsigned(binary_exponent + 14) << 10 |
                                      unsigned(std::llround((mantissa - 1) * 1024)));
}

bool check(cudaError_t error) {
    if (error == cudaSuccess) { return true; }
    std::cerr << cudaGetErrorString(error) << '\n';
    return false;
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    std::vector<std::uint8_t> codes(kCases * 8);
    for (int i = 0; i < kCases; ++i) {
        for (int pair = 0; pair < 8; ++pair) {
            // Every byte appears in every position; different words expose shift/index defects.
            codes[i * 8 + pair] = static_cast<std::uint8_t>((i % 256) + 37 * pair);
        }
    }
    std::uint8_t* device_codes = nullptr;
    std::uint16_t* out8 = nullptr;
    std::uint16_t* out16 = nullptr;
    const std::size_t output_bytes = kCases * 16 * sizeof(std::uint16_t);
    if (!check(cudaMalloc(&device_codes, codes.size())) ||
        !check(cudaMalloc(&out8, output_bytes)) || !check(cudaMalloc(&out16, output_bytes)) ||
        !check(cudaMemcpy(device_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice))) {
        return 1;
    }
    decode<<<(kCases + 127) / 128, 128>>>(device_codes, out8, out16);
    std::vector<std::uint16_t> host8(kCases * 16), host16(kCases * 16);
    const bool copied = check(cudaGetLastError()) &&
        check(cudaMemcpy(host8.data(), out8, output_bytes, cudaMemcpyDeviceToHost)) &&
        check(cudaMemcpy(host16.data(), out16, output_bytes, cudaMemcpyDeviceToHost));
    cudaFree(device_codes);
    cudaFree(out8);
    cudaFree(out16);
    if (!copied) { return 1; }
    for (int i = 0; i < kCases; ++i) {
        for (int lane = 0; lane < 16; ++lane) {
            const unsigned nibble = (codes[i * 8 + lane / 2] >> (4 * (lane % 2))) & 15;
            const auto expected = oracle(nibble, i / 256);
            if (host8[i * 16 + lane] != expected || host16[i * 16 + lane] != expected) {
                std::cerr << "FAIL scale=" << i / 256 << " byte=" << i % 256
                          << " lane=" << lane << " expected=" << expected
                          << " f16x8=" << host8[i * 16 + lane]
                          << " f16x16=" << host16[i * 16 + lane] << '\n';
                return 1;
            }
        }
    }
    std::cout << "OK NVFP4 KV exact FP16 decode: 256 bytes x 127 legal scales x 16 lanes, "
                 "f16x8 and f16x16\n";
    return 0;
}
