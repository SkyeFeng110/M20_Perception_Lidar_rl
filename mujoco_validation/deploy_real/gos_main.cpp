// GOS-only entry for the copied 1516 runtime. No ONNX packaging or motor API.
#include "gos.hpp"
#include "aos.hpp" // Reuse the original board-local tmux ancestry check.
using namespace m20deploy;
int main(int argc,char** argv) {
    try {
        if(argc<2 || std::string(argv[1])=="--help") {
            std::cout << "m20_deploy gos-run|stop|check-session [--option value ...]\n";
            return argc<2 ? 2 : 0;
        }
        Options options(argc,argv);
        const std::string mode=argv[1];
        if(mode=="gos-run") gos_run(options);
        else if(mode=="stop") owned_stop(options);
        else if(mode=="check-session") {
            options.finish(); require(durable_session(),"Board-local tmux required");
            std::cout << "GOS_DURABLE_SESSION=PASS scope=no_joint_commands\n";
        } else throw std::runtime_error("Unsupported GOS operation: "+mode);
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "M20_DEPLOY=FAIL " << error.what() << '\n'; return 1;
    }
}
