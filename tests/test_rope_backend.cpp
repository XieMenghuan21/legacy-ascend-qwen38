#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr int64_t dimension = 256;

std::vector<float> reference(const std::vector<float> & data, const std::vector<int32_t> & position,
                             int64_t heads, int64_t tokens, int64_t row, float freq_scale,
                             float attn_factor) {
    std::vector<float> result(static_cast<size_t>(dimension * heads * tokens));
    const float theta_scale = std::pow(10000000.0f, -2.0f / 64.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t head = 0; head < heads; ++head) {
            const size_t input_offset = static_cast<size_t>((token * heads + head) * row);
            const size_t output_offset = static_cast<size_t>((token * heads + head) * dimension);
            std::copy_n(data.begin() + input_offset, dimension, result.begin() + output_offset);
            // Match ggml_mrope_cache_init's repeated F32 multiplication order.
            float theta[4] = { static_cast<float>(position[token]), static_cast<float>(position[token + tokens]),
                               static_cast<float>(position[token + 2 * tokens]), static_cast<float>(position[token + 3 * tokens]) };
            for (int index = 0; index < 32; ++index) {
                const int axis = index % 3 == 1 && index < 33 ? 1 :
                    index % 3 == 2 && index < 30 ? 2 :
                    index % 3 == 0 && index < 33 ? 0 : 3;
                const float angle = freq_scale * theta[axis];
                const float cosine = std::cos(angle) * attn_factor;
                const float sine = std::sin(angle) * attn_factor;
                const float lo = data[input_offset + index];
                const float hi = data[input_offset + index + 32];
                result[output_offset + index] = lo * cosine - hi * sine;
                result[output_offset + index + 32] = lo * sine + hi * cosine;
                for (float & value : theta) value *= theta_scale;
            }
        }
    }
    return result;
}

void run_case(ggml_backend_t backend, int64_t heads, int64_t tokens, bool padded) {
    context_ptr context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!context) throw std::runtime_error("ggml_init failed");
    const int64_t row = padded ? dimension * 2 : dimension;
    ggml_tensor * storage = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, row, heads, tokens);
    ggml_tensor * x = padded ? ggml_view_3d(context.get(), storage, dimension, heads, tokens,
        static_cast<size_t>(row) * sizeof(float), static_cast<size_t>(row * heads) * sizeof(float), 0) : storage;
    ggml_tensor * p = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 4 * tokens);
    int sections[4] = {11, 11, 10, 0};
    ggml_tensor * output = ggml_rope_multi(context.get(), x, p, nullptr, 64, sections,
        GGML_ROPE_TYPE_IMROPE, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_set_input(storage);
    ggml_set_input(p);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("RoPE is not supported by selected backend");
    buffer_ptr allocation(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!allocation) throw std::runtime_error("Backend allocation failed");
    std::vector<float> data(static_cast<size_t>(row * heads * tokens));
    for (size_t index = 0; index < data.size(); ++index) {
        data[index] = std::sin(static_cast<float>(index) * 0.117f) * 1.9f;
    }
    ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(float));
    std::vector<int32_t> position(static_cast<size_t>(4 * tokens));
    std::vector<float> actual(static_cast<size_t>(dimension * heads * tokens));
    // Deliberately distinguish all 4 position axes and change position across
    // repeated graphs. This detects accidental standard-RoPE and stale caches.
    for (const int32_t base : {0, 17, 127, 8191}) {
        for (int64_t token = 0; token < tokens; ++token) {
            for (int axis = 0; axis < 4; ++axis) {
                position[static_cast<size_t>(axis * tokens + token)] = base + static_cast<int32_t>(token) + axis * 3;
            }
        }
        ggml_backend_tensor_set(p, position.data(), 0, position.size() * sizeof(int32_t));
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Graph compute failed");
        ggml_backend_synchronize(backend);
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        const auto expected = reference(data, position, heads, tokens, row, 1.0f, 1.0f);
        double max_abs = 0;
        double squared_error = 0;
        double squared_reference = 0;
        size_t tail_errors = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index])) throw std::runtime_error("Nonfinite output");
            const double error = std::abs(static_cast<double>(actual[index]) - expected[index]);
            max_abs = std::max(max_abs, error);
            squared_error += error * error;
            squared_reference += static_cast<double>(expected[index]) * expected[index];
            tail_errors += index % dimension >= 64 && actual[index] != expected[index];
        }
        const double nrmse = std::sqrt(squared_error / squared_reference);
        // At long positions CPU recurrence and NPU exponent-then-position may
        // differ by several ulps in phase; also report every measured error.
        const double abs_limit = base < 1000 ? 0.0002 : 0.008;
        const double nrmse_limit = base < 1000 ? 0.00002 : 0.0008;
        const bool passed = max_abs <= abs_limit && nrmse <= nrmse_limit && tail_errors == 0;
        std::cout << "heads=" << heads << " tokens=" << tokens << " padded=" << padded
                  << " base=" << base << " max_abs=" << max_abs << " nrmse=" << nrmse
                  << " tail_errors=" << tail_errors << " elapsed_us=" << elapsed
                  << " status=" << (passed ? "PASS" : "FAIL") << std::endl;
        if (!passed) throw std::runtime_error("RoPE numerical acceptance failed");
    }
}
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: test_rope_backend BACKEND_NAME BACKEND_LIBRARY_DIR");
        ggml_backend_load_all_from_path(argv[2]);
        backend_ptr backend(ggml_backend_init_by_name(argv[1], nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("Backend initialization failed");
        for (const int64_t heads : {4, 24}) {
            for (const int64_t tokens : {1, 23}) {
                for (const bool padded : {false, true}) run_case(backend.get(), heads, tokens, padded);
            }
        }
        std::cout << "All 32 IMRoPE checks passed" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
