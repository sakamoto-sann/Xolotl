// Optional host-mapped embedding restoration for the pinned 1017 Q2_K model.
#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace strata::v41 {
inline bool validate_embedding_sidecar(const strata::GgufModel & side,
                                      const std::vector<uint64_t> & expected,
                                      std::string & error) {
    auto reject=[&](const std::string & why){error="embedding sidecar: "+why;return false;};
    auto meta=[&](const char * key,const char * value){const auto * v=side.meta().get(key);return v&&v->s==value;};
    if(side.size()!=1 || !meta("general.architecture","deepseek41-embedding") ||
       !meta("deepseek41.embedding.source_repo","deepseek-ai/DeepSeek-V4.1-Flash-1017") ||
       !meta("deepseek41.embedding.source_revision","1bae0e1088993b9b82de08f59de5be5a34e42170") ||
       !meta("deepseek41.embedding.source_tensor","embed.weight") ||
       !meta("deepseek41.embedding.source_shard_sha256","43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7"))
        return reject("unsupported source or architecture");
    if(expected.size()!=2 || !expected[0] || !expected[1] || side.shard(0).tensors().size()!=1)
        return reject("unexpected tensor count or expected shape");
    const auto & t=side.shard(0).tensors().front();
    // GGML_TYPE_BF16 = 30. Retain the original bits, mapped on the host; no whole-table GPU allocation.
    if(t.name!="token_embd.weight" || t.type!=30 || t.shape!=expected || !side.in_bounds(t,0))
        return reject("name, type, shape or bounds mismatch");
    const uint8_t * data=side.shard(0).tensor_data(t);
    for(uint64_t i=0;i<t.elements();++i) {
        uint16_t bits;std::memcpy(&bits,data+i*sizeof(bits),sizeof(bits));
        if((bits&0x7f80u)==0x7f80u)return reject("nonfinite BF16 value");
    }
    return true;
}
} // namespace strata::v41
