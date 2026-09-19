#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t K = 64;
constexpr int64_t N = 16;
constexpr int64_t M = 8;
constexpr int ROUNDS = 10;
constexpr double ABSOLUTE_TOLERANCE = 1e-3;

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

class matrix_case {
public:
    explicit matrix_case(int device)
        : device_(device), name_("CANN" + std::to_string(device)),
          backend_(ggml_backend_init_by_name(name_.c_str(), nullptr), ggml_backend_free),
          context_(nullptr, ggml_free), buffer_(nullptr, ggml_backend_buffer_free),
          weights_(static_cast<size_t>(K * N)), input_(static_cast<size_t>(K * M)),
          actual_(static_cast<size_t>(N * M)) {
        if (!backend_) {
            throw std::runtime_error("Cannot initialize required backend " + name_);
        }
        const ggml_init_params parameters{2 * 1024 * 1024, nullptr, true};
        context_.reset(ggml_init(parameters));
        if (!context_) {
            throw std::runtime_error("ggml_init failed for " + name_);
        }
        weight_tensor_ = ggml_new_tensor_2d(context_.get(), GGML_TYPE_F16, K, N);
        input_tensor_ = ggml_new_tensor_2d(context_.get(), GGML_TYPE_F32, K, M);
        ggml_set_input(weight_tensor_);
        ggml_set_input(input_tensor_);
        ggml_set_name(weight_tensor_, "mm_cache_weights");
        ggml_set_name(input_tensor_, "mm_cache_input");
        // Use GGML's default tensor formats and multiplication attributes.
        result_ = ggml_mul_mat(context_.get(), weight_tensor_, input_tensor_);
        ggml_set_name(result_, "mm_cache_result");
        ggml_set_output(result_);
        if (result_->type != GGML_TYPE_F32 || result_->ne[0] != N || result_->ne[1] != M ||
            result_->ne[2] != 1 || result_->ne[3] != 1 || !ggml_is_contiguous(result_) ||
            ggml_nbytes(result_) != actual_.size() * sizeof(float)) {
            throw std::runtime_error("Unexpected MUL_MAT output layout on " + name_);
        }
        graph_ = ggml_new_graph_custom(context_.get(), 16, false);
        ggml_build_forward_expand(graph_, result_);
        for (int index = 0; index < ggml_graph_n_nodes(graph_); ++index) {
            ggml_tensor * node = ggml_graph_node(graph_, index);
            if (!ggml_backend_supports_op(backend_.get(), node)) {
                throw std::runtime_error(name_ + " does not support " + ggml_op_name(node->op));
            }
        }
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend_.get()));
        if (!buffer_) {
            throw std::runtime_error("Cannot allocate independent graph buffer on " + name_);
        }
        ggml_backend_buffer_clear(buffer_.get(), 0);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                // Each device gets different small integers, all exact in FP16.
                const float value = static_cast<float>((5 * k + 3 * n + device_) % 7 - 3);
                weights_[static_cast<size_t>(n * K + k)] = ggml_fp32_to_fp16(value);
            }
        }
        ggml_backend_tensor_set(weight_tensor_, weights_.data(), 0, weights_.size() * sizeof(ggml_fp16_t));
        ggml_backend_synchronize(backend_.get());
        std::cout << "initialized=" << name_ << " actual_backend=" << ggml_backend_name(backend_.get())
                  << " weight=F16[64,16] input=F32[64,8] output=F32[16,8]"
                  << " buffer_bytes=" << ggml_backend_buffer_get_size(buffer_.get()) << '\n';
    }

    matrix_case(const matrix_case &) = delete;
    matrix_case & operator=(const matrix_case &) = delete;

    bool run(int round) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                // Refresh inputs each execution to expose stale cached addresses.
                input_[static_cast<size_t>(m * K + k)] =
                    static_cast<float>((3 * k + 5 * m + 2 * device_ + 7 * round) % 9 - 4);
            }
        }
        ggml_backend_tensor_set(input_tensor_, input_.data(), 0, input_.size() * sizeof(float));
        const auto begin = std::chrono::steady_clock::now();
        // Direct execution on the selected backend cannot silently fall back to CPU.
        const ggml_status status = ggml_backend_graph_compute(backend_.get(), graph_);
        ggml_backend_synchronize(backend_.get());
        const double elapsed_us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(name_ + " graph compute failed with status " + std::to_string(status));
        }
        ggml_backend_tensor_get(result_, actual_.data(), 0, actual_.size() * sizeof(float));

        size_t failed = 0;
        size_t worst_index = 0;
        double max_absolute = 0;
        double worst_expected = 0;
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                double expected = 0;
                for (int64_t k = 0; k < K; ++k) {
                    const double weight = static_cast<double>(
                        ggml_fp16_to_fp32(weights_[static_cast<size_t>(n * K + k)]));
                    expected += weight * static_cast<double>(input_[static_cast<size_t>(m * K + k)]);
                }
                const size_t index = static_cast<size_t>(m * N + n);
                const double error = std::isfinite(actual_[index])
                    ? std::abs(static_cast<double>(actual_[index]) - expected)
                    : std::numeric_limits<double>::infinity();
                if (error > max_absolute) {
                    max_absolute = error;
                    worst_index = index;
                    worst_expected = expected;
                }
                if (error > ABSOLUTE_TOLERANCE) {
                    ++failed;
                }
            }
        }
        std::cout << std::setprecision(9) << "round=" << round + 1 << " backend=" << name_
                  << " graph_us=" << elapsed_us << " max_abs=" << max_absolute
                  << " failed_elements=" << failed << '/' << actual_.size()
                  << " result=" << (failed == 0 ? "PASS" : "FAIL") << '\n';
        if (failed != 0) {
            std::cerr << "worst_index=" << worst_index << " expected=" << worst_expected
                      << " actual=" << actual_[worst_index] << " tolerance=" << ABSOLUTE_TOLERANCE << '\n';
        }
        return failed == 0;
    }

private:
    int device_;
    std::string name_;
    // Reverse destruction releases the graph buffer before context and backend.
    backend_ptr backend_;
    context_ptr context_;
    buffer_ptr buffer_;
    ggml_tensor * weight_tensor_ = nullptr;
    ggml_tensor * input_tensor_ = nullptr;
    ggml_tensor * result_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::vector<ggml_fp16_t> weights_;
    std::vector<float> input_;
    std::vector<float> actual_;
};
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 2) {
            throw std::runtime_error("Usage: test_mm_cache [BACKEND_LIBRARY_DIR]");
        }
        if (argc == 2) {
            ggml_backend_load_all_from_path(argv[1]);
        } else {
            ggml_backend_load_all();
        }
        // Both graphs remain allocated throughout the alternating execution loop.
        matrix_case card4(4);
        matrix_case card5(5);
        const std::array<matrix_case *, 2> cases{&card4, &card5};
        for (int round = 0; round < ROUNDS; ++round) {
            if (!cases[static_cast<size_t>(round) % cases.size()]->run(round)) {
                return 2;
            }
        }
        std::cout << "PASS alternating_executions=" << ROUNDS
                  << " backends=CANN4,CANN5 absolute_tolerance=" << ABSOLUTE_TOLERANCE << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
