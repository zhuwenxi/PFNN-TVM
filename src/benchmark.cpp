#include <pfnn_benchmark.hpp>
#include <cstdio>
#include <cstdlib>
#include "argagg.hpp"

int main(int argc, char* argv[]) {

    std::string model_dir_path = "./models";
    std::string lib_path = "./lib.so";
    int batch = 5;
    int repeats = 200;

    argagg::parser argparser {{
        { "help", {"-h", "--help"},
            "shows this help message", 0},
            { "batch", {"-b", "--batch"},
                "Batch size (default: 5)", 1},
            { "model", {"-m", "--model"},
                "Model directory path (default: ./models)", 1},
            { "repeats", {"-r", "--repeats"},
                "Repeat tests for benchmark (default: 200)", 1},
            { "library", {"-l", "--lib"},
                "TVM exported dynamic library path (default: ./lib.so)", 1},
    }};

    argagg::parser_results args;
    try {
        args = argparser.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    if (args["help"]) {
        std::cerr << argparser;
        return EXIT_SUCCESS;
    }
    if (args["batch"]) {
        batch = args["batch"];
    }
    if (args["model"]) {
        const char* _model_dir_path = args["model"];
        model_dir_path = std::string(_model_dir_path);
    }
    if (args["library"]) {
        const char* _lib_path = args["library"];
        lib_path = std::string(_lib_path);
    }
    if (args["repeats"]) {
        const char* _repeats = args["repeats"];
        repeats = atoi(_repeats);
    }

    PFNN_Benchmark *benchmark = new PFNN_Benchmark(batch, lib_path, 1032, 256, 908);
    benchmark->LoadModels(model_dir_path);
    benchmark->LoadSampleInputs(model_dir_path);
    benchmark->LoadSampleOutputs(model_dir_path);
    
    benchmark->BenchmarkInference(repeats);
    benchmark->VerifyInference();
    delete benchmark;

    return 0;
}
