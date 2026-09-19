#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

int64_t integer(const char * text, int64_t lower, int64_t upper) {
    const std::string source(text);
    size_t consumed = 0;
    const int64_t value = std::stoll(source, &consumed);
    if (consumed != source.size() || value < lower || value > upper) {
        throw std::runtime_error("Invalid integer argument: " + source);
    }
    return value;
}

void compute(ggml_backend_t backend, ggml_cgraph * graph) {
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("RMS graph execution failed");
    }
    ggml_backend_synchronize(backend);
}

void run(ggml_backend_t backend, int64_t width, int64_t rows, int64_t stride, float epsilon) {
    context_ptr context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!context) {
        throw std::runtime_error("ggml_init failed");
    }
    ggml_tensor * storage = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, stride, rows);
    ggml_set_input(storage);
    ggml_tensor * x = ggml_view_2d(context.get(), storage, width, rows, stride * sizeof(float), 0);
    ggml_tensor * result = ggml_rms_norm(context.get(), x, epsilon);
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, result);
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, index))) {
            throw std::runtime_error("Requested backend does not support RMS graph");
        }
    }
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("Backend allocation failed");
    }
    const size_t count = static_cast<size_t>(width * rows);
    std::vector<float> input(static_cast<size_t>(stride * rows));
    std::vector<float> actual(count);
    std::vector<double> reference(count);
    constexpr std::array<double, 7> SCALES{{0.0, 1.0e-8, 1.0e-4, 1.0, 100.0, 1.0e10, -1.0}};
    double largest_absolute = 0.0;
    double largest_nrmse = 0.0;

    for (size_t test = 0; test < SCALES.size(); ++test) {
        // Padded lanes contain a large sentinel to catch a wrong Q-head stride.
        std::fill(input.begin(), input.end(), 8192.0f);
        for (int64_t row = 0; row < rows; ++row) {
            double squared = 0.0;
            for (int64_t column = 0; column < width; ++column) {
                const double base = test == 6 ? 1.0 :
                    std::sin(0.017 * (column + 1) + row * 0.37) +
                    0.33 * std::cos(0.041 * column + row * 0.19);
                const float value = static_cast<float>(base * SCALES[test]);
                input[static_cast<size_t>(row * stride + column)] = value;
                squared += static_cast<double>(value) * value;
            }
            const double inverse = 1.0 / std::sqrt(squared / width + epsilon);
            for (int64_t column = 0; column < width; ++column) {
                reference[static_cast<size_t>(row * width + column)] =
                    input[static_cast<size_t>(row * stride + column)] * inverse;
            }
        }
        ggml_backend_tensor_set(storage, input.data(), 0, input.size() * sizeof(float));
        compute(backend, graph);
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        double maximum = 0.0;
        double squared_error = 0.0;
        double squared_reference = 0.0;
        size_t failed = 0;
        for (size_t index = 0; index < count; ++index) {
            if (!std::isfinite(actual[index])) {
                throw std::runtime_error("Nonfinite RMS output in case " + std::to_string(test));
            }
            const double error = std::abs(actual[index] - reference[index]);
            maximum = std::max(maximum, error);
            squared_error += error * error;
            squared_reference += reference[index] * reference[index];
            failed += error > 2.0e-5 + 2.0e-5 * std::abs(reference[index]);
        }
        const double nrmse = squared_reference == 0.0 ? std::sqrt(squared_error) :
                             std::sqrt(squared_error / squared_reference);
        largest_absolute = std::max(largest_absolute, maximum);
        largest_nrmse = std::max(largest_nrmse, nrmse);
        std::cout << "case=" << test << " scale=" << SCALES[test] << " max_abs=" << maximum
                  << " nrmse=" << nrmse << " failed_elements=" << failed << '\n';
        if (failed != 0) {
            throw std::runtime_error("RMS precision validation failed");
        }
    }
    for (int index = 0; index < 20; ++index) {
        compute(backend, graph);
    }
    const auto start = std::chrono::steady_clock::now();
    constexpr int ITERATIONS = 200;
    for (int index = 0; index < ITERATIONS; ++index) {
        compute(backend, graph);
    }
    const double microseconds = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start).count() / ITERATIONS;
    std::cout << std::setprecision(10) << "RMS_RESULT backend=" << ggml_backend_name(backend)
              << " width=" << width << " rows=" << rows << " stride=" << stride
              << " epsilon=" << epsilon << " cases=" << SCALES.size()
              << " max_abs=" << largest_absolute << " max_nrmse=" << largest_nrmse
              << " sync_graph_us=" << microseconds << " accuracy=PASS\n";
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 4 || argc > 7) {
            throw std::runtime_error("Usage: test_rms_backend WIDTH ROWS STRIDE [BACKEND=CANN4]"
                                     " [LIBRARY_DIRECTORY] [EPSILON=1e-6]");
        }
        const int64_t width = integer(argv[1], 1, 8192);
        const int64_t rows = integer(argv[2], 1, 256);
        const int64_t stride = integer(argv[3], width, 16384);
        const float epsilon = argc >= 7 ? std::stof(argv[6]) : 1.0e-6f;
        if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
            throw std::runtime_error("EPSILON must be finite and positive");
        }
        if (argc >= 6) {
            ggml_backend_load_all_from_path(argv[5]);
        } else {
            ggml_backend_load_all();
        }
        const char * name = argc >= 5 ? argv[4] : "CANN4";
        backend_ptr backend(ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
        if (!backend) {
            throw std::runtime_error(std::string("Cannot initialize backend: ") + name);
        }
        run(backend.get(), width, rows, stride, epsilon);
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
