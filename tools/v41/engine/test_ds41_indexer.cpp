// CUDA long-context indexer regression: packed MXFP4 vs BF16 and sampled float64 reference.
#include "ds41_ops.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace strata::v41;
int main() {
    auto backend=ggml_backend_cuda_init(0);if(!backend)return 2;
    int failures=0;constexpr int D=128,H=64;
    const float grid[]={0,.5f,1,1.5f,2,3,4,6,-0.0f,-.5f,-1,-1.5f,-2,-3,-4,-6};
    for(int64_t rows : {8192LL,65537LL,131072LL})for(int queries : {1,3,16}) {
        ggml_init_params params{ggml_tensor_overhead()*128+ggml_graph_overhead(),nullptr,true};
        auto ctx=ggml_init(params);if(!ctx)return 2;
        auto input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,D,rows);
        auto ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,rows);
        auto q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,H,queries,1);
        auto w=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,H,queries,1,1);
        auto mask=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,rows,queries,1,1);
        for(auto t : {input,ids,q,w,mask}){ggml_set_input(t);ggml_set_output(t);}
        auto rounded=ggml_dsv41_fake_quant(ctx,input,(int)Ds41Quant::FP4_POW2_32);
        auto packed=ggml_set_rows(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_MXFP4,D,rows),rounded,ids);
        auto bf16=ggml_set_rows(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,D,rows),rounded,ids);
        auto a=ggml_lightning_indexer(ctx,q,ggml_reshape_4d(ctx,bf16,D,1,rows,1),w,mask);
        auto b=ggml_lightning_indexer(ctx,q,ggml_reshape_4d(ctx,packed,D,1,rows,1),w,mask);
        ggml_set_output(a);ggml_set_output(b);
        auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,a);ggml_build_forward_expand(graph,b);
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if(!ggml_gallocr_alloc_graph(alloc,graph))return 2;
        std::vector<float> keys((size_t)rows*D),qs((size_t)D*H*queries),weights((size_t)H*queries);
        std::vector<int32_t> indices((size_t)rows);std::vector<ggml_fp16_t> masks((size_t)rows*queries,ggml_fp32_to_fp16(0));
        for(int64_t row=0;row<rows;++row){indices[(size_t)row]=(int32_t)row;for(int i=0;i<D;++i){
            float v=grid[(row*17+i*13)%16]*std::ldexp(1.0f,int((row+i/32)%9)-4);
            if(row==rows-2)v=1536;if(row==rows-1)v=3072;keys[(size_t)row*D+i]=v;
        }}
        for(int t=0;t<queries;++t){masks[(size_t)t*rows+rows-1]=ggml_fp32_to_fp16(-INFINITY);
            for(int h=0;h<H;++h){weights[(size_t)t*H+h]=float(1+h%4)/256;
                for(int i=0;i<D;++i)qs[((size_t)t*H+h)*D+i]=.5f+float((h*7+i*3+t)%8)*.125f;
            }}
        ggml_backend_tensor_set(input,keys.data(),0,keys.size()*4);ggml_backend_tensor_set(ids,indices.data(),0,indices.size()*4);
        ggml_backend_tensor_set(q,qs.data(),0,qs.size()*4);ggml_backend_tensor_set(w,weights.data(),0,weights.size()*4);
        ggml_backend_tensor_set(mask,masks.data(),0,masks.size()*sizeof(ggml_fp16_t));
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 2;
        std::vector<float> av((size_t)rows*queries),bv(av.size());
        ggml_backend_tensor_get(a,av.data(),0,av.size()*4);ggml_backend_tensor_get(b,bv.data(),0,bv.size()*4);
        size_t bit_bad=0,ref_bad=0,best_bad=0;
        for(size_t i=0;i<av.size();++i)bit_bad+=std::memcmp(&av[i],&bv[i],4)!=0;
        std::vector<int64_t> sample{0,1,31,32,32767,32768,65535,65536,rows-2,rows-1};
        std::sort(sample.begin(),sample.end());sample.erase(std::unique(sample.begin(),sample.end()),sample.end());
        for(int t=0;t<queries;++t){
            auto first=bv.begin()+(size_t)t*rows;best_bad+=(std::max_element(first,first+rows)-first)!=rows-2;
            for(auto row:sample){if(row>=rows)continue;
                double ref=0;
                if(row==rows-1)ref=-INFINITY;
                else for(int h=0;h<H;++h){double dot=0;for(int i=0;i<D;++i)dot+=(double)qs[((size_t)t*H+h)*D+i]*keys[(size_t)row*D+i];ref+=std::max(0.0,dot)*weights[(size_t)t*H+h];}
                const float actual=bv[(size_t)t*rows+row];
                if(std::isinf(ref)){ref_bad+=!(std::isinf(actual)&&actual<0);}
                else ref_bad+=!std::isfinite(actual)||std::abs((double)actual-ref)>1e-4+std::abs(ref)*1e-5;
            }
        }
        std::printf("indexer rows=%lld queries=%d: packed_vs_bf16_bit_bad=%zu sampled_f64_bad=%zu known_best_bad=%zu\n",(long long)rows,queries,bit_bad,ref_bad,best_bad);
        failures+=bit_bad||ref_bad||best_bad;
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    ggml_backend_free(backend);return failures?1:0;
}
