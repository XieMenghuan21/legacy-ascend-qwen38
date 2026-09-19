#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t WIDTH = 128;
constexpr int64_t KEY_HEADS = 16;
constexpr double ATOL = 2e-6;
constexpr double RTOL = 2e-5;

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

std::vector<float> read_floats(const std::string & directory, const std::string & name, size_t count) {
    const std::string path = directory + "/" + name + ".f32.bin";
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(count * sizeof(float))) {
        throw std::runtime_error("Missing fixture or unexpected byte size: " + path);
    }
    std::vector<float> values(count);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
        throw std::runtime_error("Short fixture read: " + path);
    }
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Non-finite fixture input: " + path);
    }
    return values;
}

ggml_tensor * vector_input(ggml_context * context, const char * name, int64_t heads, bool strided) {
    ggml_tensor * backing = ggml_new_tensor_4d(context, GGML_TYPE_F32, strided ? WIDTH * 2 : WIDTH, heads, 1, 1);
    ggml_set_input(backing);
    ggml_tensor * tensor = backing;
    if (strided) {
        // Keep each row contiguous while doubling the physical distance between heads.
        tensor = ggml_view_4d(context, backing, WIDTH, heads, 1, 1,
                              backing->nb[1], backing->nb[2], backing->nb[3], 0);
        ggml_set_input(tensor);
    }
    ggml_set_name(tensor, name);
    return tensor;
}

void set_vector(ggml_tensor * tensor, const std::vector<float> & values) {
    const size_t row_elements = static_cast<size_t>(tensor->ne[0]);
    for (int64_t head = 0; head < tensor->ne[1]; ++head) {
        ggml_backend_tensor_set(tensor, values.data() + head * row_elements,
                                head * tensor->nb[1], row_elements * sizeof(float));
    }
}

struct difference {
    double max_absolute = 0;
    double max_scaled = 0;
    size_t failed_elements = 0;
    size_t max_absolute_index = 0;
};

difference compare(const float * actual, const std::vector<float> & expected) {
    difference result;
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual[index])) {
            ++result.failed_elements;
            result.max_absolute = INFINITY;
            result.max_scaled = INFINITY;
            continue;
        }
        const double error = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double tolerance = ATOL + RTOL * std::abs(static_cast<double>(expected[index]));
        if (error > result.max_absolute) {
            result.max_absolute = error;
            result.max_absolute_index = index;
        }
        result.max_scaled = std::max(result.max_scaled, error / tolerance);
        if (error > tolerance) {
            ++result.failed_elements;
        }
    }
    return result;
}

void print_difference(const char * name, const difference & result) {
    std::cout << name << ": max_abs=" << result.max_absolute
              << ", max_scaled_error=" << result.max_scaled
              << ", failed_elements=" << result.failed_elements
              << ", max_abs_index=" << result.max_absolute_index << '\n';
}

bool run_case(ggml_backend_t backend, const std::string & directory, bool strided, int64_t value_heads) {
    const size_t output_floats = static_cast<size_t>(value_heads) * WIDTH;
    const size_t state_floats = output_floats * WIDTH;
    const ggml_init_params parameters{4 * 1024 * 1024, nullptr, true};
    context_ptr context(ggml_init(parameters), ggml_free);
    if (!context) {
        throw std::runtime_error("ggml_init failed");
    }
    ggml_tensor * q = vector_input(context.get(), "q", KEY_HEADS, strided);
    ggml_tensor * k = vector_input(context.get(), "k", KEY_HEADS, strided);
    ggml_tensor * v = vector_input(context.get(), "v", value_heads, strided);
    ggml_tensor * g = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, 1, value_heads, 1, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, 1, value_heads, 1, 1);
    ggml_tensor * state = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, WIDTH, WIDTH, value_heads, 1);
    for (ggml_tensor * input : {g, beta, state}) {
        ggml_set_input(input);
    }
    ggml_set_name(g, "g_log");
    ggml_set_name(beta, "beta");
    ggml_set_name(state, "state_vk");
    // Fixtures already contain L2-normalized Q/K. Do not normalize or scale again.
    ggml_tensor * result = ggml_gated_delta_net(context.get(), q, k, v, g, beta, state, 1);
    ggml_set_name(result, "gdn_result");
    ggml_set_output(result);
    if (ggml_nbytes(result) != (output_floats + state_floats) * sizeof(float)) {
        throw std::runtime_error("Unexpected GGML GDN output ABI");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, result);
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        ggml_tensor * node = ggml_graph_node(graph, index);
        if (!ggml_backend_supports_op(backend, node)) {
            throw std::runtime_error(std::string("Backend does not support graph node: ") + ggml_op_name(node->op));
        }
    }
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("Backend tensor allocation failed");
    }
    ggml_backend_buffer_clear(buffer.get(), 0);
    set_vector(q, read_floats(directory, "q", KEY_HEADS * WIDTH));
    set_vector(k, read_floats(directory, "k", KEY_HEADS * WIDTH));
    set_vector(v, read_floats(directory, "v", output_floats));
    const auto gates = read_floats(directory, "g_log", value_heads);
    const auto betas = read_floats(directory, "beta", value_heads);
    const auto initial_state = read_floats(directory, "state_vk", state_floats);
    const auto expected_output = read_floats(directory, "expected_output", output_floats);
    const auto expected_state = read_floats(directory, "expected_state_vk", state_floats);
    ggml_backend_tensor_set(g, gates.data(), 0, gates.size() * sizeof(float));
    ggml_backend_tensor_set(beta, betas.data(), 0, betas.size() * sizeof(float));
    ggml_backend_tensor_set(state, initial_state.data(), 0, initial_state.size() * sizeof(float));
    // Direct backend execution cannot silently reschedule this op to CPU.
    const auto begin = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const double elapsed_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Backend graph compute failed with status " + std::to_string(status));
    }
    std::vector<float> actual(output_floats + state_floats);
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    const difference output_error = compare(actual.data(), expected_output);
    const difference state_error = compare(actual.data() + output_floats, expected_state);
    const bool passed = output_error.failed_elements == 0 && state_error.failed_elements == 0;
    std::cout << std::setprecision(9)
              << "backend=" << ggml_backend_name(backend)
              << ", layout=" << (strided ? "strided-heads" : "contiguous")
              << ", Hq=16,Hk=16,Hv=" << value_heads << ",D=128,T=1,B=1,K=1"
              << ", first_graph_us=" << elapsed_us << '\n';
    print_difference("output", output_error);
    print_difference("state", state_error);
    std::cout << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 5) {
            throw std::runtime_error("Usage: test_gdn_backend FIXTURE_DIR [BACKEND=CANN4] [BACKEND_LIBRARY_DIR] [VALUE_HEADS=32|48]");
        }
        if (argc >= 4) {
            ggml_backend_load_all_from_path(argv[3]);
        } else {
            ggml_backend_load_all();
        }
        int64_t value_heads = 32;
        if (argc == 5) {
            const std::string argument = argv[4];
            if (argument != "32" && argument != "48") {
                throw std::runtime_error("VALUE_HEADS must be 32 or 48");
            }
            value_heads = argument == "48" ? 48 : 32;
        }
        const char * name = argc >= 3 ? argv[2] : "CANN4";
        backend_ptr backend(ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
        if (!backend) {
            throw std::runtime_error(std::string("Cannot initialize requested backend: ") + name);
        }
        const bool contiguous_passed = run_case(backend.get(), argv[1], false, value_heads);
        const bool strided_passed = run_case(backend.get(), argv[1], true, value_heads);
        return contiguous_passed && strided_passed ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
