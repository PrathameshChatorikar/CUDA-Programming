#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input.cpp" << std::endl;
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Failed to open " << argv[1] << std::endl;
        return 1;
    }

    std::unordered_map<std::string, std::string> copy_map;
    std::string line;

    while (std::getline(infile, line)) {
        std::string output = line;

        // Simple pattern match: int a = b;
        std::size_t pos = line.find("int ");
        if (pos != std::string::npos) {
            std::size_t eq_pos = line.find("=");
            std::size_t semi_pos = line.find(";");
            if (eq_pos != std::string::npos && semi_pos != std::string::npos) {
                std::string lhs = line.substr(pos + 4, eq_pos - (pos + 4));
                std::string rhs = line.substr(eq_pos + 1, semi_pos - (eq_pos + 1));
                // Trim whitespace
                lhs.erase(0, lhs.find_first_not_of(" \t"));
                lhs.erase(lhs.find_last_not_of(" \t") + 1);
                rhs.erase(0, rhs.find_first_not_of(" \t"));
                rhs.erase(rhs.find_last_not_of(" \t") + 1);

                // Register copy
                copy_map[lhs] = rhs;
            }
        } else {
            // Replace known copies
            for (const auto& entry : copy_map) {
                std::size_t var_pos = output.find(entry.first);
                while (var_pos != std::string::npos) {
                    output.replace(var_pos, entry.first.length(), entry.second);
                    var_pos = output.find(entry.first, var_pos + entry.second.length());
                }
            }
        }

        // Output transformed line
        std::cout << output << std::endl;
    }

    return 0;
}
