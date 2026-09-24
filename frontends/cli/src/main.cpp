// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <openframegen/core/runtime.hpp>

#include <iostream>
#include <string_view>

namespace {

constexpr std::string_view kBanner = R"(   ___                   ______                          ______
  / _ \ _ __   ___ _ __ |  ___| __ __ _ _ __ ___   ___/ ___| ___ _ __
 | | | | '_ \ / _ \ '_ \| |_ | '__/ _` | '_ ` _ \ / _ \ |  _ / _ \ '_ \
 | |_| | |_) |  __/ | | |  _|| | | (_| | | | | | |  __/ |_| |  __/ | | |
  \___/| .__/ \___|_| |_|_|  |_|  \__,_|_| |_| |_|\___|\____|\___|_| |_|
       |_|

             Open Frame Generation
)";

void print_help() {
    std::cout
        << kBanner << '\n'
        << "Usage: ofg <command>\n\n"
        << "Commands:\n"
        << "  info       Show runtime and build information\n"
        << "  version    Show the OpenFrameGen version\n"
        << "  help       Show this help message\n\n"
        << "The frame-generation backends are not implemented yet.\n";
}

void print_info() {
    const auto info = ofg::runtime_info();

    std::cout
        << kBanner << '\n'
        << "Version:   " << info.version << '\n'
        << "Platform:  " << info.platform_name << '\n'
        << "Frontend:  CLI\n"
        << "Core:      ready\n"
        << "GPU:       not initialized\n"
        << "FrameGen:  not implemented\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help();
        return 0;
    }

    const std::string_view command{argv[1]};

    if (command == "info") {
        print_info();
        return 0;
    }

    if (command == "version" || command == "--version" || command == "-v") {
        std::cout << "OpenFrameGen " << ofg::version() << '\n';
        return 0;
    }

    if (command == "help" || command == "--help" || command == "-h") {
        print_help();
        return 0;
    }

    std::cerr << "Unknown command: " << command << "\n\n";
    print_help();
    return 1;
}
