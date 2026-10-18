#include "ds41_embedding_sidecar.hpp"
#include <cstdio>
int main(int argc,char ** argv) {
    if(argc<3)return 2;
    try {
        // Tiny fixture dimensions are fixed by the test, not inferred from the supplied file.
        const std::vector<uint64_t> expected{256,4};
        auto good=strata::GgufModel::open(argv[1]);std::string error;
        if(!strata::v41::validate_embedding_sidecar(good,expected,error)) {
            std::fprintf(stderr,"%s\n",error.c_str());return 1;
        }
        for(int i=2;i<argc;++i) {
            bool rejected=false;
            try {auto bad=strata::GgufModel::open(argv[i]);rejected=!strata::v41::validate_embedding_sidecar(bad,expected,error);}
            catch(const std::exception &) {rejected=true;}
            if(!rejected) {std::fprintf(stderr,"invalid fixture accepted: %d\n",i);return 1;}
        }
        std::puts("embedding sidecar: valid accepted; all invalid fixtures rejected");return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
