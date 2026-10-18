// Synchronized CUDA graph wall time, not kernel-only time. No model or weights required.
#include "ds41_ops.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace strata::v41;
struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ~Graph() { if (alloc) ggml_gallocr_free(alloc); if (ctx) ggml_free(ctx); }
    bool init(ggml_backend_t backend, int64_t n, Ds41Quant mode, bool fused, const std::vector<float> & x) {
        ggml_init_params params{ggml_tensor_overhead()*512 + ggml_graph_overhead(), nullptr, true};
        ctx=ggml_init(params); if (!ctx) return false;
        input=ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, n/512); ggml_set_input(input); ggml_set_output(input);
        output=fused?ggml_dsv41_fake_quant(ctx,input,(int)mode):ds41_fake_quant(ctx,input,mode);
        ggml_set_output(output); graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,output);
        alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc,graph)) return false;
        ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float)); return true;
    }
};
static double time_graph(ggml_backend_t backend, Graph & g, int iterations) {
    ggml_backend_synchronize(backend);
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;++i) if(ggml_backend_graph_compute(backend,g.graph)!=GGML_STATUS_SUCCESS) return -1;
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
}
int main() {
    auto backend=ggml_backend_cuda_init(0); if(!backend) return 2;
    int failures=0;
    for(int64_t n : {512LL,4096LL,16384LL,131072LL,2097152LL}) {
        std::vector<float> x((size_t)n);
        for(size_t i=0;i<x.size();++i) x[i]=float(int((i*73+17)%4093)-2046)/127.0f;
        for(auto mode : {Ds41Quant::FP8_POW2_32,Ds41Quant::FP4_E4M3_16,Ds41Quant::FP4_POW2_32}) {
            Graph ref,fused;
            if(!ref.init(backend,n,mode,false,x)||!fused.init(backend,n,mode,true,x)) return 2;
            if(time_graph(backend,ref,10)<0||time_graph(backend,fused,10)<0) return 2;
            std::vector<float> a((size_t)n),b((size_t)n);
            ggml_backend_tensor_get(ref.output,a.data(),0,a.size()*4);ggml_backend_tensor_get(fused.output,b.data(),0,b.size()*4);
            size_t bad=0;for(size_t i=0;i<a.size();++i)bad+=std::memcmp(&a[i],&b[i],4)!=0;
            failures+=bad!=0;
            std::vector<double> rt,ft; const int iterations=n>=2097152?30:200;
            for(int repeat=0;repeat<5;++repeat) {
                if(repeat%2) {ft.push_back(time_graph(backend,fused,iterations));rt.push_back(time_graph(backend,ref,iterations));}
                else {rt.push_back(time_graph(backend,ref,iterations));ft.push_back(time_graph(backend,fused,iterations));}
            }
            if (*std::min_element(rt.begin(),rt.end())<0 || *std::min_element(ft.begin(),ft.end())<0) return 2;
            std::sort(rt.begin(),rt.end());std::sort(ft.begin(),ft.end());
            std::printf("BENCH {\"n\":%lld,\"mode\":%d,\"iterations\":%d,\"repeats\":5,\"composite_us\":%.4f,\"fused_us\":%.4f,\"bit_mismatches\":%zu}\n",(long long)n,(int)mode,iterations,rt[2],ft[2],bad);
        }
    }
    ggml_backend_free(backend);return failures?1:0;
}
