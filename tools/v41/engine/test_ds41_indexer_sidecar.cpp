#include "ds41_indexer_sidecar.hpp"
#include <cstdio>
int main(int argc,char ** argv){
    if(argc<3)return 2;
    try {
        auto valid=strata::GgufModel::open(argv[1]);
        strata::v41::IndexerShapes expected;
        for(const auto & t:valid.shard(0).tensors())expected.emplace(t.name,t.shape);
        std::string error;
        if(!strata::v41::validate_indexer_sidecar(valid,expected,error)){std::fprintf(stderr,"%s\n",error.c_str());return 1;}
        for(int i=2;i<argc;++i){
            bool rejected=false;
            try{auto bad=strata::GgufModel::open(argv[i]);rejected=!strata::v41::validate_indexer_sidecar(bad,expected,error);}
            catch(const std::exception &){rejected=true;}
            if(!rejected){std::fprintf(stderr,"invalid sidecar accepted: argument %d\n",i);return 1;}
        }
        std::printf("indexer sidecar: valid accepted; %d invalid fixtures rejected\n",argc-2);return 0;
    } catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
