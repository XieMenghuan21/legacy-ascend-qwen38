// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Project contributors
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
constexpr std::string_view HELP =
    "Usage: model-benchmark --model PATH --devices ID[,ID...]\n"
    "       (--prompt TEXT | --regression) [--context 256]\n"
    "       [--output-tokens 64] [--threads 8]\n\n"
    "Serial greedy Qwen text generation with the thinking prefix closed.\n"
    "Device IDs are explicit CANN backend IDs; there is no default device.\n"
    "Each request emits one JSON record on stdout. Library logs use stderr.\n"
    "--regression runs arithmetic, Chinese QA, Python code, then arithmetic again.\n"
    "--help returns before loading backends or a model.\n";

struct options {
    std::string model;
    std::vector<int32_t> devices;
    std::string prompt;
    int32_t context = 256;
    int32_t output_tokens = 64;
    int32_t threads = 8;
    bool regression = false;
};

int32_t number(std::string_view text, int32_t minimum, const std::string & option) {
    int32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < minimum) {
        throw std::runtime_error(option + " expects an integer >= " + std::to_string(minimum));
    }
    return value;
}

std::vector<int32_t> device_ids(const std::string & text) {
    std::vector<int32_t> result;
    std::set<int32_t> seen;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = comma == std::string::npos ? text.size() : comma;
        const int32_t id = number(std::string_view(text).substr(begin, end - begin), 0, "--devices");
        if (!seen.insert(id).second) {
            throw std::runtime_error("--devices contains duplicate ID " + std::to_string(id));
        }
        result.push_back(id);
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return result;
}

options parse(int argc, char ** argv) {
    options value;
    std::set<std::string> supplied;
    for (int index = 1; index < argc; ++index) {
        const std::string key(argv[index]);
        if (!supplied.insert(key).second) {
            throw std::runtime_error("Duplicate option: " + key);
        }
        if (key == "--regression") {
            value.regression = true;
            continue;
        }
        if (key != "--model" && key != "--devices" && key != "--prompt" &&
            key != "--context" && key != "--output-tokens" && key != "--threads") {
            throw std::runtime_error("Unknown option: " + key);
        }
        if (++index >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        const std::string argument(argv[index]);
        if (key == "--model") value.model = argument;
        if (key == "--devices") value.devices = device_ids(argument);
        if (key == "--prompt") value.prompt = argument;
        if (key == "--context") value.context = number(argument, 1, key);
        if (key == "--output-tokens") value.output_tokens = number(argument, 1, key);
        if (key == "--threads") value.threads = number(argument, 1, key);
    }
    if (value.model.empty() || value.devices.empty()) {
        throw std::runtime_error("--model and --devices are required");
    }
    if (value.regression == (supplied.count("--prompt") != 0)) {
        throw std::runtime_error("Choose exactly one of --prompt or --regression");
    }
    if (!value.regression && value.prompt.empty()) {
        throw std::runtime_error("--prompt must not be empty");
    }
    if (value.output_tokens >= value.context) {
        throw std::runtime_error("--output-tokens must leave space for the prompt in --context");
    }
    return value;
}

struct json_text {
    std::string encoded;
    bool valid_utf8 = true;
};

json_text quote_json(std::string_view text) {
    constexpr char HEX[] = "0123456789abcdef";
    json_text result{"\"", true};
    for (size_t index = 0; index < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        if (byte < 0x80) {
            if (byte == '"' || byte == '\\') {
                result.encoded += '\\';
                result.encoded += static_cast<char>(byte);
            } else if (byte < 0x20) {
                result.encoded += "\\u00";
                result.encoded += HEX[byte >> 4];
                result.encoded += HEX[byte & 0x0f];
            } else {
                result.encoded += static_cast<char>(byte);
            }
            ++index;
            continue;
        }
        const size_t length = byte >= 0xc2 && byte <= 0xdf ? 2 :
                              byte >= 0xe0 && byte <= 0xef ? 3 :
                              byte >= 0xf0 && byte <= 0xf4 ? 4 : 0;
        bool valid = length != 0 && length <= text.size() - index;
        if (valid) {
            for (size_t offset = 1; offset < length; ++offset) {
                const unsigned char continuation = static_cast<unsigned char>(text[index + offset]);
                valid = valid && continuation >= 0x80 && continuation <= 0xbf;
            }
            const unsigned char second = static_cast<unsigned char>(text[index + 1]);
            valid = valid && !(byte == 0xe0 && second < 0xa0) && !(byte == 0xed && second >= 0xa0) &&
                    !(byte == 0xf0 && second < 0x90) && !(byte == 0xf4 && second >= 0x90);
        }
        if (valid) {
            result.encoded.append(text.substr(index, length));
            index += length;
        } else {
            // A token-budget cut can split a UTF-8 sequence; emit valid JSON and report it.
            result.encoded += "\\ufffd";
            result.valid_utf8 = false;
            ++index;
        }
    }
    result.encoded += '"';
    return result;
}

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

struct backend_lifetime {
    backend_lifetime() {
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~backend_lifetime() { llama_backend_free(); }
    backend_lifetime(const backend_lifetime &) = delete;
    backend_lifetime & operator=(const backend_lifetime &) = delete;
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & user_text, int32_t maximum_tokens) {
    const std::string text = "<|im_start|>user\n" + user_text +
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Prompt exceeds tokenizer text-length limit");
    }
    const int32_t bytes = static_cast<int32_t>(text.size());
    const int32_t needed = llama_tokenize(vocab, text.data(), bytes, nullptr, 0, true, true);
    if (needed >= 0 || needed == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenizer returned an invalid prompt length");
    }
    if (-needed > maximum_tokens) {
        throw std::runtime_error("Prompt tokens (" + std::to_string(-needed) +
                                 ") plus output budget exceed --context");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-needed));
    const int32_t written = llama_tokenize(vocab, text.data(), bytes, tokens.data(), -needed, true, true);
    if (written <= 0 || written > -needed) {
        throw std::runtime_error("Prompt tokenization failed");
    }
    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::array<char, 512> stack{};
    int32_t size = llama_token_to_piece(vocab, token, stack.data(), static_cast<int32_t>(stack.size()), 0, true);
    if (size >= 0) {
        return std::string(stack.data(), static_cast<size_t>(size));
    }
    if (size == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Token piece length overflow");
    }
    std::string result(static_cast<size_t>(-size), '\0');
    size = llama_token_to_piece(vocab, token, result.data(), -size, 0, true);
    if (size < 0 || static_cast<size_t>(size) > result.size()) {
        throw std::runtime_error("Token piece conversion failed");
    }
    result.resize(static_cast<size_t>(size));
    return result;
}

void decode(llama_context * context, llama_token * tokens, int32_t count) {
    const int32_t status = llama_decode(context, llama_batch_get_one(tokens, count));
    if (status != 0) {
        throw std::runtime_error("Decode failed with status " + std::to_string(status));
    }
}

bool regression_passed(size_t index, const std::string & answer) {
    if (index == 0 || index == 3) {
        const size_t first = answer.find_first_not_of(" \t\r\n");
        const size_t last = answer.find_last_not_of(" \t\r\n");
        return first != std::string::npos && answer.substr(first, last - first + 1) == "391";
    }
    if (index == 1) {
        return answer.find("北京") != std::string::npos;
    }
    return answer.find("def ") != std::string::npos && answer.find("return") != std::string::npos;
}

void json_number(double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}

bool request(llama_context * context, const llama_vocab * vocab, const options & config,
             size_t index, std::vector<llama_token> & tokens) {
    llama_memory_t memory = llama_get_memory(context);
    if (memory == nullptr) {
        throw std::runtime_error("Model context has no resettable sequence memory");
    }
    llama_memory_clear(memory, true);
    llama_perf_context_reset(context);
    sampler_ptr sampler(llama_sampler_init_greedy(), llama_sampler_free);
    if (!sampler) {
        throw std::runtime_error("Greedy sampler initialization failed");
    }
    for (size_t offset = 0; offset < tokens.size();) {
        const size_t remaining = tokens.size() - offset;
        size_t count = std::min<size_t>(64, remaining);
        // Avoid classifying a one-token prompt tail as decode in llama's counters.
        if (remaining > 64 && remaining - count == 1) {
            --count;
        }
        decode(context, tokens.data() + offset, static_cast<int32_t>(count));
        offset += count;
    }
    std::string answer;
    int32_t produced = 0;
    int32_t sampled = 0;
    bool ended = false;
    for (int32_t step = 0; step < config.output_tokens; ++step) {
        llama_token next = llama_sampler_sample(sampler.get(), context, -1);
        ++sampled;
        if (llama_vocab_is_eog(vocab, next)) {
            ended = true;
            break;
        }
        answer += token_piece(vocab, next);
        ++produced;
        if (step + 1 < config.output_tokens) {
            decode(context, &next, 1);
        }
    }
    llama_synchronize(context);
    const llama_perf_context_data performance = llama_perf_context(context);
    const json_text answer_json = quote_json(answer);
    const bool passed = !config.regression || regression_passed(index, answer);
    std::cout << std::setprecision(12) << "{\"request_index\":" << index
              << ",\"answer\":" << answer_json.encoded
              << ",\"answer_utf8_valid\":" << (answer_json.valid_utf8 ? "true" : "false")
              << ",\"prompt_tokens\":" << tokens.size()
              << ",\"output_tokens\":" << produced
              << ",\"sampled_tokens\":" << sampled
              << ",\"requested_output_tokens\":" << config.output_tokens
              << ",\"context_tokens\":" << config.context
              << ",\"stop_reason\":\"" << (ended ? "eos" : "length")
              << "\",\"n_p_eval\":" << performance.n_p_eval
              << ",\"n_eval\":" << performance.n_eval << ",\"t_p_eval_ms\":";
    json_number(performance.t_p_eval_ms);
    std::cout << ",\"t_eval_ms\":";
    json_number(performance.t_eval_ms);
    std::cout << ",\"t_load_ms\":";
    json_number(performance.t_load_ms);
    std::cout << ",\"decode_tokens_per_second\":";
    if (performance.t_eval_ms > 0.0 && performance.n_eval > 0) {
        json_number(1000.0 * performance.n_eval / performance.t_eval_ms);
    } else {
        std::cout << "null";
    }
    std::cout << ",\"regression_passed\":";
    if (config.regression) std::cout << (passed ? "true" : "false");
    else std::cout << "null";
    std::cout << "}\n" << std::flush;
    if (!std::cout) {
        throw std::runtime_error("Cannot write benchmark JSON to stdout");
    }
    return passed;
}

int run(const options & config) {
    backend_lifetime backend;
    std::vector<ggml_backend_dev_t> devices;
    devices.reserve(config.devices.size() + 1);
    for (int32_t id : config.devices) {
        const std::string name = "CANN" + std::to_string(id);
        ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
        if (device == nullptr) {
            throw std::runtime_error("Requested backend device is unavailable: " + name);
        }
        devices.push_back(device);
    }
    devices.push_back(nullptr);
    llama_model_params model_parameters = llama_model_default_params();
    model_parameters.devices = devices.data();
    model_parameters.n_gpu_layers = 99;
    model_ptr model(llama_model_load_from_file(config.model.c_str(), model_parameters), llama_model_free);
    if (!model) {
        throw std::runtime_error("Model loading failed");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("Model has no vocabulary");
    }
    const std::vector<std::string> prompts = config.regression ? std::vector<std::string>{
        "计算17乘23，只输出结果。", "请只回答中国的首都，不要解释。",
        "只给出一个Python回文判断函数的代码，不要解释和示例。", "计算17乘23，只输出结果。"
    } : std::vector<std::string>{config.prompt};
    std::vector<std::vector<llama_token>> requests;
    requests.reserve(prompts.size());
    for (const std::string & prompt : prompts) {
        auto tokens = tokenize(vocab, prompt, config.context - config.output_tokens);
        if (tokens.size() + static_cast<size_t>(config.output_tokens) > static_cast<size_t>(config.context)) {
            throw std::runtime_error("Prompt tokens (" + std::to_string(tokens.size()) +
                                     ") plus output budget exceed --context");
        }
        requests.push_back(std::move(tokens));
    }
    llama_context_params context_parameters = llama_context_default_params();
    context_parameters.n_ctx = static_cast<uint32_t>(config.context);
    context_parameters.n_batch = std::min<uint32_t>(64, context_parameters.n_ctx);
    context_parameters.n_ubatch = std::min<uint32_t>(32, context_parameters.n_batch);
    context_parameters.n_seq_max = 1;
    context_parameters.n_threads = config.threads;
    context_parameters.n_threads_batch = config.threads;
    context_parameters.op_offload = true;
    context_parameters.no_perf = false;
    context_parameters.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_ptr context(llama_init_from_model(model.get(), context_parameters), llama_free);
    if (!context) {
        throw std::runtime_error("Context initialization failed");
    }
    bool passed = true;
    for (size_t index = 0; index < requests.size(); ++index) {
        passed = request(context.get(), vocab, config, index, requests[index]) && passed;
    }
    return passed ? 0 : 2;
}
}  // namespace

int main(int argc, char ** argv) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--help") {
            std::cout << HELP;
            return 0;
        }
    }
    try {
        return run(parse(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
