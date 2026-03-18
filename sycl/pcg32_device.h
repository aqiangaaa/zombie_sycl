#pragma once

// ============================================================
// 设备侧 PCG32 随机数生成器
//
// 从 deps/pcg32/pcg32.h 精简而来，去掉所有 host-only 依赖
// （std::algorithm, std::cmath, assert 等），
// 保留纯算术实现，可在 SYCL kernel 中安全使用。
//
// 参考：Melissa O'Neill, http://www.pcg-random.org
// ============================================================

#include <cstdint>

#define PCG32_MULT 0x5851f42d4c957f2dULL

struct PCG32State {
    std::uint64_t state;
    std::uint64_t inc;
};

// 从 (initstate, initseq) 初始化 PCG32 状态
// 与原始 pcg32::seed() 完全一致
inline PCG32State pcg32_seed(std::uint64_t initstate,
                              std::uint64_t initseq = 1u) {
    PCG32State rng;
    rng.state = 0u;
    rng.inc = (initseq << 1u) | 1u;

    // 第一次 advance
    rng.state = rng.state * PCG32_MULT + rng.inc;
    rng.state += initstate;
    // 第二次 advance
    rng.state = rng.state * PCG32_MULT + rng.inc;

    return rng;
}

// 生成一个 uint32 随机数并推进状态
inline std::uint32_t pcg32_next(PCG32State& rng) {
    std::uint64_t oldstate = rng.state;
    rng.state = oldstate * PCG32_MULT + rng.inc;
    std::uint32_t xorshifted =
        static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    std::uint32_t rot = static_cast<std::uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

// 生成 [0, 1) 均匀浮点数
inline float pcg32_next_float(PCG32State& rng) {
    return static_cast<float>(pcg32_next(rng)) / 4294967296.0f;
}
