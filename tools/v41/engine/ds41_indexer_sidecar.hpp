// Opt-in 1017 indexer sidecar validation, independent of backend allocation.
#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace strata::v41 {
using IndexerShapes = std::map<std::string,std::vector<uint64_t>>;
inline bool validate_indexer_sidecar(const strata::GgufModel & side, const IndexerShapes & expected,
                                     std::string & error) {
    auto reject=[&](const std::string & message){error="indexer sidecar: "+message;return false;};
    auto meta=[&](const char * name,const char * value){const auto * v=side.meta().get(name);return v&&v->s==value;};
    const bool query_input=meta("general.architecture","deepseek41-retrieval-qinput");
    const bool combined=query_input || meta("general.architecture","deepseek41-retrieval");
    if(side.size()!=1 || (!combined && !meta("general.architecture","deepseek41-indexer")) ||
       !meta("deepseek41.indexer.source_repo","deepseek-ai/DeepSeek-V4.1-Flash-1017") ||
       !meta("deepseek41.indexer.source_revision","1bae0e1088993b9b82de08f59de5be5a34e42170"))
        return reject("unsupported source or architecture");
    if(expected.size()!=(query_input?43u:combined?35u:24u) || side.shard(0).tensors().size()!=expected.size())
        return reject("unexpected retrieval tensor count");
    std::set<std::string> seen;
    for(const auto & t:side.shard(0).tensors()) {
        if(!seen.insert(t.name).second)return reject("duplicate tensor "+t.name);
        const auto it=expected.find(t.name);
        if(it==expected.end())return reject("unexpected tensor "+t.name);
        if(t.type!=0 || t.shape!=it->second || !side.in_bounds(t,0))
            return reject("type, shape or bounds mismatch for "+t.name);
        const uint8_t * data=side.shard(0).tensor_data(t);
        for(uint64_t i=0;i<t.elements();++i) {
            float value;std::memcpy(&value,data+i*sizeof(float),sizeof(float));
            if(!std::isfinite(value))return reject("nonfinite tensor "+t.name);
        }
    }
    return true;
}
} // namespace strata::v41
