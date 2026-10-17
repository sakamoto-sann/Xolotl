// Included inside strata::v41 by ds41_geometry.hpp; no backend or model data reads.
// Opt-in contract: mrfakename/DeepSeek-V4.1-Flash-1017-GGUF Q2_K, revision
// 2c97dba3934f27a99767a90830e7519cc7b8b2d9. Absent fields are pinned to official
// config.json (1bae0e1088993b9b82de08f59de5be5a34e42170), never inferred from another model.
inline std::string ds41_llama_name(std::string n) {
    for (const auto & p : std::initializer_list<std::pair<const char *, const char *>>{
        {"engram_q_norm.weight", "engram_q.weight"}, {"engram_k_norm.weight", "engram_k.weight"},
        {"engram_kv.weight", "engram_wkv.weight"}}) {
        const size_t pos = n.find(p.first);
        if (pos != std::string::npos) n.replace(pos, std::strlen(p.first), p.second);
    }
    return n;
}
inline std::string ds41_native_name(std::string n) {
    for (const auto & p : std::initializer_list<std::pair<const char *, const char *>>{
        {"engram_q.weight", "engram_q_norm.weight"}, {"engram_k.weight", "engram_k_norm.weight"},
        {"engram_wkv.weight", "engram_kv.weight"}}) {
        const size_t pos = n.find(p.first);
        if (pos != std::string::npos) n.replace(pos, std::strlen(p.first), p.second);
    }
    return n;
}
inline bool ds41_llama_table(const std::string & n) {
    return n.find(".engram_embd.weight") != std::string::npos;
}
inline std::string ds41_read_llama_geometry(const GgufModel & model, Ds41Geometry & G) {
    const auto & g = model.meta();
    const auto * arch = g.get("general.architecture");
    if (!arch || arch->s != "deepseek41" || g.get("deepseek41.config"))
        return "--llama-q2-k requires a llama.cpp deepseek41 GGUF";
    G.llama_q2_k = true;
    std::string e;
    const std::string P = "deepseek41.";
    if (!(e = detail::u64(g, P + "embedding_length", G.n_embd)).empty()) return e;
    if (G.n_embd != 5120) return "unsupported 1017 embedding_length";
    if (!(e = detail::u64(g, P + "block_count", G.n_layer)).empty()) return e;
    if (G.n_layer != 40) return "unsupported 1017 block_count";
    if (!(e = detail::u64(g, P + "attention.head_count", G.n_head)).empty()) return e;
    if (G.n_head != 64) return "unsupported 1017 attention.head_count";
    if (!(e = detail::u64(g, P + "attention.key_length", G.d_head)).empty()) return e;
    if (G.d_head != 512) return "unsupported 1017 attention.key_length";
    if (!(e = detail::u64(g, P + "rope.dimension_count", G.d_rope)).empty()) return e;
    if (G.d_rope != 64) return "unsupported 1017 rope.dimension_count";
    if (!(e = detail::u64(g, P + "attention.q_lora_rank", G.r_q)).empty()) return e;
    if (G.r_q != 1280) return "unsupported 1017 attention.q_lora_rank";
    if (!(e = detail::u64(g, P + "attention.output_lora_rank", G.o_lora)).empty()) return e;
    if (G.o_lora != 1024) return "unsupported 1017 attention.output_lora_rank";
    if (!(e = detail::u64(g, P + "attention.output_group_count", G.o_groups)).empty()) return e;
    if (G.o_groups != 8) return "unsupported 1017 attention.output_group_count";
    if (!(e = detail::u64(g, P + "attention.sliding_window", G.n_swa)).empty()) return e;
    if (G.n_swa != 128) return "unsupported 1017 attention.sliding_window";
    if (!(e = detail::u64(g, P + "expert_count", G.n_expert)).empty()) return e;
    if (G.n_expert != 384) return "unsupported 1017 expert_count";
    if (!(e = detail::u64(g, P + "expert_used_count", G.n_used)).empty()) return e;
    if (G.n_used != 6) return "unsupported 1017 expert_used_count";
    if (!(e = detail::u64(g, P + "expert_feed_forward_length", G.n_ff)).empty()) return e;
    if (G.n_ff != 2304) return "unsupported 1017 expert_feed_forward_length";
    if (!(e = detail::u64(g, P + "expert_shared_count", G.n_shared)).empty()) return e;
    if (G.n_shared != 1) return "unsupported 1017 expert_shared_count";
    if (!(e = detail::u64(g, P + "hyper_connection.count", G.hc)).empty()) return e;
    if (G.hc != 4) return "unsupported 1017 hyper_connection.count";
    if (!(e = detail::u64(g, P + "hyper_connection.sinkhorn_iterations", G.hc_iters)).empty()) return e;
    if (G.hc_iters != 20) return "unsupported 1017 hyper_connection.sinkhorn_iterations";
    if (!(e = detail::u64(g, P + "attention.indexer.head_count", G.idx_heads)).empty()) return e;
    if (G.idx_heads != 32) return "unsupported 1017 attention.indexer.head_count";
    if (!(e = detail::u64(g, P + "attention.indexer.key_length", G.idx_dim)).empty()) return e;
    if (G.idx_dim != 128) return "unsupported 1017 attention.indexer.key_length";
    if (!(e = detail::u64(g, P + "attention.indexer.top_k", G.idx_topk)).empty()) return e;
    if (G.idx_topk != 512) return "unsupported 1017 attention.indexer.top_k";
    if (!(e = detail::u64(g, P + "context_length", G.ctx_len)).empty()) return e;
    if (G.ctx_len != 1048576) return "unsupported 1017 context_length";
    if (!(e = detail::u64(g, P + "engram.head_count", G.engram_heads)).empty()) return e;
    if (G.engram_heads != 8) return "unsupported 1017 engram.head_count";
    if (!(e = detail::u64(g, P + "engram.key_length", G.engram_dim)).empty()) return e;
    if (G.engram_dim != 256) return "unsupported 1017 engram.key_length";
    if (!(e = detail::u64(g, P + "engram.max_ngram_size", G.engram_ngram)).empty()) return e;
    if (G.engram_ngram != 4) return "unsupported 1017 engram.max_ngram_size";
    if (!(e = detail::u64(g, P + "engram.pad_id", G.engram_pad)).empty()) return e;
    if (G.engram_pad != 2) return "unsupported 1017 engram.pad_id";
    if (!(e = detail::f64(g, P + "expert_weights_scale", G.weights_scale)).empty()) return e;
    if (!std::isfinite(G.weights_scale) || std::fabs(G.weights_scale - 1.5) > 1.5e-05) return "unsupported 1017 expert_weights_scale";
    if (!(e = detail::f64(g, P + "hyper_connection.epsilon", G.hc_eps)).empty()) return e;
    if (!std::isfinite(G.hc_eps) || std::fabs(G.hc_eps - 1e-06) > 1e-11) return "unsupported 1017 hyper_connection.epsilon";
    if (!(e = detail::f64(g, P + "attention.layer_norm_rms_epsilon", G.rms_eps)).empty()) return e;
    if (!std::isfinite(G.rms_eps) || std::fabs(G.rms_eps - 1e-20) > 1e-25) return "unsupported 1017 attention.layer_norm_rms_epsilon";
    if (!(e = detail::f64(g, P + "rope.freq_base", G.rope_base)).empty()) return e;
    if (!std::isfinite(G.rope_base) || std::fabs(G.rope_base - 10000) > 0.1) return "unsupported 1017 rope.freq_base";
    if (!(e = detail::f64(g, P + "attention.compress_rope_freq_base", G.compress_rope_base)).empty()) return e;
    if (!std::isfinite(G.compress_rope_base) || std::fabs(G.compress_rope_base - 160000) > 1.6) return "unsupported 1017 attention.compress_rope_freq_base";
    if (!(e = detail::f64(g, P + "rope.scaling.factor", G.rope_factor)).empty()) return e;
    if (!std::isfinite(G.rope_factor) || std::fabs(G.rope_factor - 16) > 0.00016) return "unsupported 1017 rope.scaling.factor";
    if (!(e = detail::f64(g, P + "rope.scaling.yarn_beta_fast", G.beta_fast)).empty()) return e;
    if (!std::isfinite(G.beta_fast) || std::fabs(G.beta_fast - 32) > 0.00032) return "unsupported 1017 rope.scaling.yarn_beta_fast";
    if (!(e = detail::f64(g, P + "rope.scaling.yarn_beta_slow", G.beta_slow)).empty()) return e;
    if (!std::isfinite(G.beta_slow) || std::fabs(G.beta_slow - 1) > 1e-05) return "unsupported 1017 rope.scaling.yarn_beta_slow";
    if (!(e = detail::f64(g, P + "rope.scaling.original_context_length", G.orig_ctx)).empty()) return e;
    if (!std::isfinite(G.orig_ctx) || std::fabs(G.orig_ctx - 65536) > 0.65536) return "unsupported 1017 rope.scaling.original_context_length";
    const auto * norm = g.get(P + "expert_weights_norm");
    const auto * gate = g.get(P + "expert_gating_func");
    if (!norm || norm->u != 1 || !gate || gate->num() != 4) return "unsupported 1017 routing";
    G.norm_topk = true;
    G.vocab = 129280; G.cand_layer = 20; G.cand_blocks = 2048; G.cand_bsize = 8;
    G.kv_sources = {2, 8, 14, 20}; G.index_sources = {2, 8, 14, 20, 24, 28, 32, 36};
    G.engram_cvocab = 99092; G.engram_rows = {384006168, 384016682};
    std::vector<int64_t> ratios;
    if (!(e = detail::arr(g, P + "attention.compress_ratios", ratios)).empty()) return e;
    if (ratios.size() != 43) return "1017 compress_ratios must include 40 main + 3 draft layers";
    for (size_t l = 0; l < ratios.size(); ++l)
        if (ratios[l] != (l < 2 || l >= 40 ? 0 : l < 20 ? 2 : 1)) return "unsupported 1017 compression schedule";
    G.ratio.assign(ratios.begin(), ratios.begin() + 40);
    for (const char * key : {"swiglu_clamp_exp", "swiglu_clamp_shexp"}) {
        const auto * v = g.get(P + key);
        if (!v || v->type != MetaType::ARRAY || v->count != 40 || v->items.size() != 40)
            return "missing 1017 clamp array";
        for (const auto & x : v->items) if (x.num() != 10) return "unsupported 1017 clamp";
    }
    G.swiglu_limit = 10;
    if (!(e = detail::arr(g, P + "engram.layer_ids", G.engram_layers)).empty()) return e;
    if (G.engram_layers != std::vector<int64_t>{1, 14}) return "unsupported 1017 Engram layers";
    std::vector<int64_t> tm, primes, offsets;
    if (!(e = ds41_read_int_array(g.path(), P + "engram.token_map", tm)).empty()) return e;
    if (!(e = ds41_read_int_array(g.path(), P + "engram.primes", primes)).empty()) return e;
    if (!(e = ds41_read_int_array(g.path(), P + "engram.multipliers", G.engram_mult)).empty()) return e;
    if (!(e = ds41_read_int_array(g.path(), P + "engram.offsets", offsets)).empty()) return e;
    if (tm.size() != (size_t) G.vocab || primes.size() != 48 || offsets.size() != 48 || G.engram_mult.size() != 8)
        return "unsupported 1017 ngram arrays";
    for (int64_t x : tm) if (x < 0 || x >= G.engram_cvocab) return "1017 token map out of range";
    for (size_t l = 0; l < 2; ++l) {
        int64_t sum = 0;
        for (size_t j = 0; j < 24; ++j) {
            const size_t i = l * 24 + j;
            if (primes[i] <= 0 || primes[i] > G.engram_rows[l] || offsets[i] != sum) return "invalid 1017 ngram buckets";
            sum += primes[i];
        }
        if (sum != G.engram_rows[l]) return "1017 ngram rows mismatch";
    }
    G.engram_token_map.assign(tm.begin(), tm.end()); G.engram_primes.assign(primes.begin(), primes.end());
    if (!(e = detail::u64(g, "tokenizer.ggml.bos_token_id", G.bos)).empty()) return e;
    if (!(e = detail::u64(g, "tokenizer.ggml.eos_token_id", G.eos)).empty()) return e;
    return {};
}
