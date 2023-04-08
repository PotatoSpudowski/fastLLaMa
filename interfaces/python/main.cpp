#include "c/main.cpp"

// #include <pybind11/embed.h>
// #include <pybind11/functional.h>
// #include <pybind11/stl.h>
// #include "logger.hpp"
// #include "bridge.hpp"
// #include <memory>

// namespace py = pybind11;
// namespace fl = fastllama;

// struct PyFastLlama {
//     PyFastLlama(fl::FastLlama&& model)
//         : inner(std::move(model))
//     {}
//     PyFastLlama(PyFastLlama const&) = delete;
//     PyFastLlama(PyFastLlama &&) = default;
//     PyFastLlama& operator=(PyFastLlama const&) = delete;
//     PyFastLlama& operator=(PyFastLlama &&) = default;
//     ~PyFastLlama() = default;
//     fl::FastLlama inner;
// };

// using LoggerFunction = typename fl::DefaultLogger::LoggerFunction;
// using LoggerResetFunction = typename fl::DefaultLogger::LoggerResetFunction;
// using PyLogFunction = std::function<void(std::string const&, std::string const&)>;

// inline static auto make_py_logger_func(PyLogFunction&& fn) {
//     return [fn = std::move(fn)](char const* func_name, int func_name_size, char const* message, int message_size) {
//         fn(std::string(func_name, static_cast<std::size_t>(func_name_size)), std::string(message, static_cast<std::size_t>(message_size)));
//     };
// }

// inline static PyLogFunction make_def_info_logger_func() {
//     return [](std::string const& func_name, std::string message) {
//         printf("\x1b[32;1m[Info]:\x1b[0m \x1b[32mFunc('%.*s') %.*s\x1b[0m", static_cast<int>(func_name.size()), func_name.data(), static_cast<int>(message.size()), message.data());
//         fflush(stdout);
//     };
// }

// inline static PyLogFunction make_def_err_logger_func() {
//     return [](std::string const& func_name, std::string message) {
//         fprintf(stderr, "\x1b[31;1m[Error]:\x1b[0m \x1b[31mFunc('%.*s') %.*s\x1b[0m", static_cast<int>(func_name.size()), func_name.data(), static_cast<int>(message.size()), message.data());
//         fflush(stdout);
//     };
// }

// inline static PyLogFunction make_def_warn_logger_func() {
//     return [](std::string const& func_name, std::string message) {
//         printf("\x1b[93;1m[Warn]:\x1b[0m \x1b[93mFunc('%.*s') %.*s\x1b[0m", static_cast<int>(func_name.size()), func_name.data(), static_cast<int>(message.size()), message.data());
//         fflush(stdout);
//     };
// }

// inline static LoggerResetFunction make_def_reset_logger_func() {
//     return []() {
//         printf("\x1b[0m");
//         fflush(stdout);
//     };
// }


// PYBIND11_MODULE(pyfastllama, m) {
//     py::class_<PyFastLlama>(m, "Model")
//         .def(py::init([](
//             std::string const& model_id,
//             std::string const& path,
//             int num_threads,
//             int n_ctx,
//             int max_tokens_in_memory,
//             int seed,
//             int tokens_to_keep,
//             int n_batch,
//             PyLogFunction log_info,
//             PyLogFunction log_err,
//             PyLogFunction log_warn,
//             LoggerResetFunction log_reset,
//             bool is_old_model
//         ) {
//             auto model_builder = fl::FastLlama::builder();
//             model_builder.seed = std::max(0, seed);
//             model_builder.n_keep = std::max(0, tokens_to_keep);
//             model_builder.n_threads = num_threads;
//             model_builder.n_ctx = n_ctx;
//             model_builder.n_batch = n_batch;
//             model_builder.is_old_model = is_old_model;
//             model_builder.last_n_tokens = static_cast<std::size_t>(max_tokens_in_memory);
            
//             auto loggerInner = fl::DefaultLogger{
//                 /*log       = */ make_py_logger_func(std::move(log_info)),
//                 /*log_err   = */ make_py_logger_func(std::move(log_err)),
//                 /*log_warn  = */ make_py_logger_func(std::move(log_warn)),
//                 /*reset     = */ std::move(log_reset)
//             };
//             model_builder.logger = fl::Logger(std::move(loggerInner));

//             auto maybe_model = model_builder.build(model_id, path);
//             if (!maybe_model) return std::unique_ptr<PyFastLlama>();
//             return std::make_unique<PyFastLlama>(std::move(*maybe_model));
//         }), py::arg("id"),
//             py::arg("path"),
//             py::arg("num_threads") = 4,
//             py::arg("n_ctx") = 512,
//             py::arg("last_n_size") = 200,
//             py::arg("seed") = 0,
//             py::arg("tokens_to_keep") = 48,
//             py::arg("n_batch") = 16,
//             py::arg("log_info") = make_def_info_logger_func(),
//             py::arg("log_err") = make_def_err_logger_func(),
//             py::arg("log_warn") = make_def_warn_logger_func(),
//             py::arg("log_reset") = make_def_reset_logger_func(),
//             py::arg("is_old_model") = false
//         ).def("ingest", [](PyFastLlama& self, std::string prompt, bool is_system_prompt = false) {
//             return self.inner.ingest(std::move(prompt), is_system_prompt);
//         }, py::arg("prompt"), py::arg("is_system_prompt") = false)
//         .def("generate", [](
//             PyFastLlama& self,
//             std::function<void(std::string const&)> fn,
//             std::size_t num_tokens,
//             int top_k,
//             float top_p,
//             float temp,
//             float repeat_penalty,
//             std::vector<std::string> const& stop_words
//         ) {
//             return self.inner.generate(std::move(fn), num_tokens, top_k, top_p, temp, repeat_penalty, stop_words);
//         },  py::arg("streaming_fn"),
//             py::arg("num_tokens") = 100,
//             py::arg("top_k") = 40,
//             py::arg("top_p") = 0.95f,
//             py::arg("temp") = 0.8f,
//             py::arg("repeat_penalty") = 1.f,
//             py::arg("stop_words") = std::vector<std::string>()
//         );
// }