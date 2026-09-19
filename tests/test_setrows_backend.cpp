#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

template<class T>
std::vector<T> read(const std::string & path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(count * sizeof(T))) {
        throw std::runtime_error("Invalid fixture: " + path);
    }
    std::vector<T> result(count);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(count * sizeof(T)))) {
        throw std::runtime_error("Cannot read fixture: " + path);
    }
    return result;
}

void run(ggml_backend_t backend, const std::string & fixtures, bool value, bool tagged) {
    context_ptr context(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!context) throw std::runtime_error("ggml_init failed");
    ggml_tensor * cache = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, value ? 1 : 1024, value ? 262144 : 256);
    ggml_tensor * x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, value ? 1 : 1024, value ? 1024 : 1);
    ggml_tensor * index = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, value ? 1024 : 1);
    ggml_tensor * output = ggml_set_rows(context.get(), cache, x, index);
    if (tagged) {
        const int32_t marker = value ? 0x53525631 : 0x53524b31;
        std::memcpy(output->op_params, &marker, sizeof(marker));
    }
    ggml_set_input(cache);
    ggml_set_input(x);
    ggml_set_input(index);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("Backend lacks SET_ROWS");
    buffer_ptr allocation(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!allocation) throw std::runtime_error("Backend allocation failed");
    const auto source = read<float>(fixtures + "/source.f32.bin", 1024);
    const auto initial = read<uint16_t>(fixtures + "/initial.f16.bin", 262144);
    ggml_backend_tensor_set(x, source.data(), 0, source.size() * sizeof(float));
    std::vector<uint16_t> actual(initial.size());
    for (const int position : {0, 1, 7, 15, 16, 17, 31, 63, 127, 128, 239, 240, 254, 255}) {
        const std::string stem = fixtures + "/" + (value ? "v" : "k") + "-p" + std::to_string(position);
        const auto indices = read<int64_t>(stem + ".i64.bin", value ? 1024 : 4);
        const auto expected = read<uint16_t>(stem + ".expected.f16.bin", 262144);
        ggml_backend_tensor_set(cache, initial.data(), 0, initial.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(index, indices.data(), 0, static_cast<size_t>(value ? 1024 : 1) * sizeof(int64_t));
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
        ggml_backend_synchronize(backend);
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(uint16_t));
        size_t mismatches = 0;
        size_t first = 0;
        for (size_t at = 0; at < actual.size(); ++at) {
            if (actual[at] != expected[at]) {
                if (mismatches == 0) first = at;
                ++mismatches;
            }
        }
        std::cout << "kind=" << (value ? "V" : "K") << " tagged=" << tagged << " position=" << position
                  << " compared_half_elements=" << actual.size() << " bit_mismatches=" << mismatches
                  << " elapsed_us=" << elapsed;
        if (mismatches != 0) std::cout << " first=" << first << " actual_bits=" << actual[first] << " expected_bits=" << expected[first];
        std::cout << std::endl;
        if (mismatches != 0) throw std::runtime_error("SET_ROWS full cache bit-exact comparison failed");
    }
}
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4 && argc != 5) throw std::runtime_error("Usage: test_setrows_backend BACKEND LIB_DIR FIXTURES [untagged]");
        if (argc == 5 && std::string(argv[4]) != "untagged") throw std::runtime_error("Last argument must be untagged");
        ggml_backend_load_all_from_path(argv[2]);
        backend_ptr backend(ggml_backend_init_by_name(argv[1], nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("Backend initialization failed");
        run(backend.get(), argv[3], false, argc == 4);
        run(backend.get(), argv[3], true, argc == 4);
        std::cout << "All 28 SET_ROWS full-cache comparisons passed" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
