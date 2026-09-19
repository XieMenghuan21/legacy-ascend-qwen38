#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t CHANNELS = 10240;
constexpr int64_t TAPS = 4;
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

std::vector<float> read_floats(const std::string & directory, const char * name, size_t count) {
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
        throw std::runtime_error("Non-finite fixture value: " + path);
    }
    return values;
}

double limit_argument(const char * text) {
    const std::string argument(text);
    size_t used = 0;
    const double value = std::stod(argument, &used);
    if (used != argument.size() || !std::isfinite(value) || value < 0) {
        throw std::runtime_error("Error limits must be finite nonnegative numbers");
    }
    return value;
}

int run(ggml_backend_t backend, const std::string & directory, int64_t tokens,
        bool use_limits, double max_abs_limit, double nrmse_limit) {
    const size_t input_count = static_cast<size_t>(tokens + TAPS - 1) * CHANNELS;
    const size_t output_count = static_cast<size_t>(tokens) * CHANNELS;
    const auto input = read_floats(directory, "input_x", input_count);
    const auto weight = read_floats(directory, "weight", CHANNELS * TAPS);
    const auto expected = read_floats(directory, "expected_output", output_count);

    const ggml_init_params parameters{2 * 1024 * 1024, nullptr, true};
    context_ptr context(ggml_init(parameters), ggml_free);
    if (!context) {
        throw std::runtime_error("ggml_init failed");
    }
    // GGML dimensions are innermost-first: x[L,C,B], weight[K,C], result[C,T,B].
    ggml_tensor * x = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, tokens + TAPS - 1, CHANNELS, 1);
    ggml_tensor * w = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, TAPS, CHANNELS);
    ggml_set_input(x);
    ggml_set_input(w);
    ggml_set_name(x, "input_x");
    ggml_set_name(w, "weight");
    ggml_tensor * result = ggml_ssm_conv(context.get(), x, w);
    ggml_set_output(result);
    ggml_set_name(result, "ssm_conv_result");
    if (result->ne[0] != CHANNELS || result->ne[1] != tokens || result->ne[2] != 1 ||
        ggml_nbytes(result) != output_count * sizeof(float)) {
        throw std::runtime_error("Unexpected GGML SSM_CONV output ABI");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 16, false);
    ggml_build_forward_expand(graph, result);
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, index))) {
            throw std::runtime_error("Requested backend does not support this graph");
        }
    }
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("Backend allocation failed");
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
    std::cout << "operator=SSM_CONV, backend=" << ggml_backend_name(backend)
              << ", C=" << CHANNELS << ", T=" << tokens << ", taps=4, B=1\n"
              << "precision_note=F32 interface and F32 reference; old-910 cubeMathType=1 permits"
                 " internal FP16 computation. F32 tensors do not prove F32 compute.\n" << std::flush;
    const auto begin = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const double first_graph_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin).count();
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Graph compute failed with status " + std::to_string(status));
    }
    const auto warm_start = std::chrono::steady_clock::now();
    for (int repeat=0;repeat<100;++repeat) {
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Warm graph failed");
        ggml_backend_synchronize(backend);
    }
    std::cout << "warm_graph_us=" << std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-warm_start).count()/100 << "\n";
    std::vector<float> actual(output_count);
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));

    size_t nonfinite = 0;
    size_t first_nonfinite_index = 0;
    size_t max_index = 0;
    size_t abs_over_1e4 = 0;
    size_t abs_over_1e3 = 0;
    double max_absolute = 0;
    double sum_absolute = 0;
    double sum_squared_error = 0;
    double sum_squared_reference = 0;
    for (size_t index = 0; index < output_count; ++index) {
        const double reference = expected[index];
        sum_squared_reference += reference * reference;
        if (!std::isfinite(actual[index])) {
            if (nonfinite == 0) {
                first_nonfinite_index = index;
            }
            ++nonfinite;
            continue;
        }
        const double error = std::abs(static_cast<double>(actual[index]) - reference);
        sum_absolute += error;
        sum_squared_error += error * error;
        abs_over_1e4 += error > 1e-4;
        abs_over_1e3 += error > 1e-3;
        if (error > max_absolute) {
            max_absolute = error;
            max_index = index;
        }
    }
    if (nonfinite > 0) {
        max_absolute = sum_absolute = sum_squared_error = std::numeric_limits<double>::infinity();
        max_index = first_nonfinite_index;
    }
    const double rmse = std::sqrt(sum_squared_error / output_count);
    const double reference_rms = std::sqrt(sum_squared_reference / output_count);
    const double normalized_rmse = reference_rms > 0 ? rmse / reference_rms :
        (rmse == 0 ? 0 : std::numeric_limits<double>::infinity());
    std::cout << std::setprecision(10)
              << "first_graph_us=" << first_graph_us << ", compared_elements=" << output_count << '\n'
              << "max_abs=" << max_absolute << ", mean_abs=" << sum_absolute / output_count
              << ", rmse=" << rmse << ", reference_rms=" << reference_rms
              << ", normalized_rmse=" << normalized_rmse << '\n'
              << "nonfinite=" << nonfinite << ", abs_over_1e-4=" << abs_over_1e4
              << ", abs_over_1e-3=" << abs_over_1e3 << '\n'
              << "max_abs_token=" << max_index / CHANNELS << ", max_abs_channel=" << max_index % CHANNELS
              << ", actual_at_max=" << actual[max_index] << ", expected_at_max=" << expected[max_index] << '\n';
    if (use_limits) {
        const bool passed = nonfinite == 0 && max_absolute <= max_abs_limit && normalized_rmse <= nrmse_limit;
        std::cout << "max_abs_limit=" << max_abs_limit << ", normalized_rmse_limit=" << nrmse_limit
                  << ", accuracy=" << (passed ? "PASS_USER_LIMITS" : "FAIL_USER_LIMITS") << '\n';
        return passed ? 0 : 2;
    }
    // Without explicit limits, exit 0 means finite execution, not precision acceptance.
    std::cout << "execution=" << (nonfinite == 0 ? "FINITE" : "NONFINITE")
              << ", accuracy=REPORT_ONLY_NO_ACCEPTANCE_LIMITS\n";
    return nonfinite == 0 ? 0 : 2;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 3 || argc > 7 || argc == 6) {
            throw std::runtime_error("Usage: test_ssm_conv_backend FIXTURE_DIR TOKENS[1|23]"
                                     " [BACKEND=CANN4] [BACKEND_LIBRARY_DIR] [MAX_ABS MAX_NRMSE]");
        }
        const std::string token_argument(argv[2]);
        if (token_argument != "1" && token_argument != "23") {
            throw std::runtime_error("TOKENS must be 1 or 23");
        }
        const int64_t tokens = token_argument == "1" ? 1 : 23;
        const bool use_limits = argc == 7;
        const double max_abs_limit = use_limits ? limit_argument(argv[5]) : 0;
        const double nrmse_limit = use_limits ? limit_argument(argv[6]) : 0;
        if (argc >= 5) {
            ggml_backend_load_all_from_path(argv[4]);
        } else {
            ggml_backend_load_all();
        }
        const char * name = argc >= 4 ? argv[3] : "CANN4";
        backend_ptr backend(ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
        if (!backend) {
            throw std::runtime_error(std::string("Cannot initialize requested backend: ") + name);
        }
        return run(backend.get(), argv[1], tokens, use_limits, max_abs_limit, nrmse_limit);
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
