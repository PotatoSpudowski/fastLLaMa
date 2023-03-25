#include "ggml.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <deque>
#include <type_traits>
#include <optional>

#include <pybind11/embed.h>
#include <pybind11/functional.h>

namespace py = pybind11;

struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;

    int n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;
};

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_model {
    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool llama_model_load(const std::string & fname, llama_model & model, gpt_vocab & vocab, int n_ctx) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    int n_ff = 0;
    int n_parts = 0;

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
        n_parts = LLAMA_N_PARTS.at(hparams.n_embd);

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_mult  = %d\n", __func__, hparams.n_mult);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
        printf("%s: n_ff    = %d\n", __func__, n_ff);
        printf("%s: n_parts = %d\n", __func__, n_parts);
    }

    // load vocab
    {
        const int32_t n_vocab = model.hparams.n_vocab;

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //if (i < 30000) {
            //    printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
            //}
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // tok_embeddings

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // output

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attention_norm

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wq
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wk
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wv
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wo

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm

        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w1
        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w2
        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w3

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (5 + 10*n_layer)*256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        model.norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        // map by name
        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;

        model.tensors["norm.weight"]   = model.norm;
        model.tensors["output.weight"] = model.output;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);

            // map by name
            model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;

            model.tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
            model.tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
            model.tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;

            model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;

            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    std::vector<uint8_t> tmp;

    for (int i = 0; i < n_parts; ++i) {
        const int part_id = i;
        //const int part_id = n_parts - i - 1;

        std::string fname_part = fname;
        if (i > 0) {
            fname_part += "." + std::to_string(i);
        }

        printf("%s: loading model part %d/%d from '%s'\n", __func__, i+1, n_parts, fname_part.c_str());

        fin = std::ifstream(fname_part, std::ios::binary);
        fin.seekg(file_offset);

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            printf("%s: ", __func__);

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ftype;

                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

                if (fin.eof()) {
                    break;
                }

                int32_t nelements = 1;
                int32_t ne[2] = { 1, 1 };
                for (int i = 0; i < n_dims; ++i) {
                    fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                std::string name(length, 0);
                fin.read(&name[0], length);

                if (model.tensors.find(name.data()) == model.tensors.end()) {
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    return false;
                }

                // split_type = 0: split by columns
                // split_type = 1: split by rows
                int split_type = 0;

                // split_type = 0:
                // regex:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = 1:
                // regex:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                if (name.find("tok_embeddings") != std::string::npos) {
                    split_type = 0;
                } else if (name.find("layers") != std::string::npos) {
                    if (name.find("attention.wo.weight") != std::string::npos) {
                        split_type = 0;
                    } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                        split_type = 0;
                    } else {
                        split_type = 1;
                    }
                } else if (name.find("output") != std::string::npos) {
                    split_type = 1;
                }

                auto tensor = model.tensors[name.data()];

                if (n_dims == 1) {
                    if (ggml_nelements(tensor) != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                } else {
                    if (ggml_nelements(tensor)/n_parts != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                }

                if (n_dims == 1) {
                    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                        fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                        return false;
                    }
                } else {
                    if (split_type == 0) {
                        if (tensor->ne[0]/n_parts != ne[0] || tensor->ne[1] != ne[1]) {
                            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0]/n_parts, tensor->ne[1], ne[0], ne[1]);
                            return false;
                        }
                    } else {
                        if (tensor->ne[0] != ne[0] || tensor->ne[1]/n_parts != ne[1]) {
                            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0], tensor->ne[1]/n_parts, ne[0], ne[1]);
                            return false;
                        }
                    }
                }

                if (0) {
                    static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                    printf("%24s - [%5d, %5d], type = %6s, split = %d\n", name.data(), ne[0], ne[1], ftype_str[ftype], split_type);
                }

                size_t bpe = 0;

                switch (ftype) {
                    case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                    case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                    case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                    case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                    default:
                            {
                                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                                return false;
                            }
                };

                if (n_dims == 1 || n_parts == 1) {
                    if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                        return false;
                    }

                    if (part_id == 0) {
                        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
                    } else {
                        fin.seekg(ggml_nbytes(tensor), std::ios::cur);
                    }

                    total_size += ggml_nbytes(tensor);
                } else {
                    if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)/n_parts) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor)/n_parts, nelements*bpe);
                        return false;
                    }

                    if (split_type == 0) {
                        const int np0 = ne[0];

                        const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                        assert(row_size == tensor->nb[1]);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = i1*row_size;
                            const size_t offset = offset_row + ((part_id*np0)/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
                        }
                    } else {
                        const int np1 = ne[1];

                        const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = (i1 + part_id*np1)*row_size;
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
                        }
                    }

                    total_size += ggml_nbytes(tensor)/n_parts;
                }

                //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
                if (++n_tensors % 8 == 0) {
                    printf(".");
                    fflush(stdout);
                }
            }

            printf(" done\n");

            printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
        }

        fin.close();
    }

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
bool llama_eval(
        const llama_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_embd/hparams.n_head;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);


            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
        }

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.norm, inpL),
                    inpL);
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.output, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

template<typename Fn>
struct FastLlamaBuffer {
    static constexpr std::size_t str_buffer_size = 512;

    FastLlamaBuffer(gpt_vocab const& vocab, std::size_t len, Fn&& fn)
        : m_vocab(vocab)
        , max_len(len)
        , m_fn(std::move(fn))
    {}

    auto push(gpt_vocab::id token) -> void {
        if (max_len <= m_buffer.size()) {
            flush_buffer();
        }
        m_buffer.push_back(token);
    }

    auto get_token_as_str(gpt_vocab::id id) const -> std::optional<std::string> {
        auto it = m_vocab.id_to_token.find(id);
        if (it != m_vocab.id_to_token.end()) return { it->second };
        return {};
    }

    auto flush_buffer() -> void {
        if (m_buffer.empty()) return;
        auto id = m_buffer.front();
        m_buffer.pop_front();
        auto temp = get_token_as_str(id);
        if (temp.has_value()) m_fn(std::move(*temp));
    }

    auto is_tokens_in_buffer(std::string_view tokens) {
        if (tokens.empty()) return std::make_pair( false, std::string_view{} );

        auto const token_to_str_view = [this](auto id) -> std::string_view {
            auto it = m_vocab.id_to_token.find(id);
            if (it != m_vocab.id_to_token.end()) return it->second;
            return "";
        };

        assert(tokens.size() < str_buffer_size && "Max token is reached");

        std::size_t buff_start = 0;

        std::for_each(m_buffer.begin(), m_buffer.end(), [token_to_str_view, &buff_start, this](auto const e) {
            assert(buff_start < str_buffer_size && "Max token is reached");
            auto temp = token_to_str_view(e);
            std::copy(temp.begin(), temp.end(), m_temp_str_buffer + buff_start);
            buff_start += temp.size();
        });

        auto temp_str = std::string_view(m_temp_str_buffer, buff_start);
        auto substr_pos = temp_str.find(tokens);
        if (substr_pos == std::string_view::npos) return std::make_pair(false, std::string_view{});
        return std::make_pair(true, temp_str.substr(0, substr_pos));
    }

private:
    gpt_vocab const& m_vocab;
    std::size_t max_len{1};
    std::deque<gpt_vocab::id> m_buffer;
    char m_temp_str_buffer[str_buffer_size];
    Fn m_fn;
};

struct FastLlama {

    FastLlama(std::string const& path, int num_threads, int n_ctx, std::size_t last_n_size, int seed)
        : m_threads(num_threads)
        , m_seed(seed)
        , m_rng(seed)
        , m_last_n_tokens(last_n_size, 0)
    {
        m_model.hparams.n_ctx = n_ctx;

        if (!llama_model_load(path, m_model, m_vocab, 512)) {
            throw std::runtime_error("Unable to load model");
        }
        if (!llama_eval(m_model, m_threads, 0, { 0, 1, 2, 3 }, m_logits, m_mem_per_token)) {
            throw std::bad_alloc();
        }
    }

    ~FastLlama() {
        ggml_free(m_model.ctx);
    }

    bool save_state(std::string const& path) {
        auto file = std::ofstream(path, std::ios::binary);
        if (!file) return false;
        save_value(file, m_seed);
        save_value(file, m_threads);
        save_value(file, n_past);
        save_value(file, m_mem_per_token);

        save_vec(file, m_embd);
        save_vec(file, m_last_n_tokens);
        save_vec(file, m_logits);

        // save_vocab(file);
        save_model(file);
        file.close();
        return true;
    }

    bool load_state(std::string const& path) {
        auto file = std::ifstream(path, std::ios::binary);
        if (!file.is_open()) return false;

        load_value(file, m_seed);
        load_value(file, m_threads);
        load_value(file, n_past);
        load_value(file, m_mem_per_token);

        m_rng.seed(m_seed);

        load_vec(file, m_embd);
        load_vec(file, m_last_n_tokens);
        load_vec(file, m_logits);
        
        // load_vocab(file);
        this->load_model(file);
        file.close();
        return true;
    }

    bool ingest(std::string const& prompt) {
        int64_t t_sample_us  = 0;
        int64_t t_predict_us = 0;

        // tokenize the prompt
        std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(m_vocab, prompt, true);

        // n_past += m_embd.size();
        // m_embd.clear();
        
        for (auto i = m_embd.size(); i < embd_inp.size(); i++) {
            // predict
            if (m_embd.size() > 0) {
                if (!llama_eval(m_model, m_threads, n_past, m_embd, m_logits, m_mem_per_token)) {
                    return false;
                }
            }

            n_past += m_embd.size();
            m_embd.clear();

            for (auto k = i; k < embd_inp.size(); k++) {
                m_embd.push_back(embd_inp[k]);
                m_last_n_tokens.erase(m_last_n_tokens.begin());
                m_last_n_tokens.push_back(embd_inp[k]);
                if (m_embd.size() > m_threads) {
                    break;
                }
            }
            i += m_embd.size() - 1;
        }

        return true;
    }

    bool generate(std::function<void(std::string const&)> fn, std::size_t num_tokens, float top_p, float temp, float repeat_penalty, std::string stop_word = nullptr) {

        auto const stop_token = stop_word.empty() ? std::vector<gpt_vocab::id>() : ::llama_tokenize(m_vocab, stop_word, false);
        // std::cout<< "Stop Word Size: " << stop_token.size() <<'\n'<<"[ ";
        // for(auto const t : stop_token) {
        //     std::cout<<std::quoted(m_vocab.id_to_token[t])<<", ";
        // }
        // std::cout<<"]";

        auto buffer = FastLlamaBuffer(m_vocab, stop_word.size() + 1, [&fn](std::string const s) {
            fn(s);
        });

        for(auto i = 0ul; i < num_tokens; ++i) {
            const int n_vocab = m_model.hparams.n_vocab;
            auto const [is_stop_token_present, to_be_flush_substr] = buffer.is_tokens_in_buffer(stop_word);
            
            if (is_stop_token_present) {
                fn(std::string(to_be_flush_substr));
                return true;
            }

            if (!llama_eval(m_model, m_threads, n_past, m_embd, m_logits, m_mem_per_token)) {
                return false;
            }

            n_past += m_embd.size();
            m_embd.clear();

            gpt_vocab::id id = 0;

            {
                id = llama_sample_top_p(m_vocab, m_logits.data() + (m_logits.size() - n_vocab), m_last_n_tokens, repeat_penalty, top_p, temp, m_rng);

                m_last_n_tokens.erase(m_last_n_tokens.begin());
                m_last_n_tokens.push_back(id);
            }

            buffer.push(id);
            m_embd.push_back(id);
        }

        if (m_embd.back() == 2) fn("[EOS]");
        
        buffer.flush_buffer();

        return true;

    }

private:

    template<typename T>
    void save_vec(std::ofstream& file, std::vector<T> const& v) const {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        save_value(file, v.size());
        file.write(reinterpret_cast<char const*>(v.data()), sizeof(T) * v.size());
    }

    void save_str(std::ofstream& file, std::string const& s) const {
        save_value(file, s.size());
        file.write(s.data(), s.size());
    }

    void save_vocab(std::ofstream& file) const {
        // printf("Save => Vocab size: %zu\n", m_vocab.id_to_token.size());
        save_value(file, m_vocab.id_to_token.size());
        for(auto const [k, v] : m_vocab.id_to_token) {
            save_value(file, k);
            save_str(file, v);
        }
    }

    template<typename T>
    void save_value(std::ofstream& f, T const& v) const {
        f.write(reinterpret_cast<char const*>(&v), sizeof(v));
    }
    
    template<typename T>
    void load_value(std::ifstream& f, T& v) const {
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
    }

    template<typename R = std::ptrdiff_t, typename T>
    R get_relative_ptr(T* other) const {
        auto* ptr = m_model.ctx->mem_buffer;
        return reinterpret_cast<R>(reinterpret_cast<std::ptrdiff_t>(other) - reinterpret_cast<std::ptrdiff_t>(ptr));
    }

    template<typename T, typename In = std::ptrdiff_t>
    T* resolve_relative_ptr(In other) const {
        auto* ptr = m_model.ctx->mem_buffer;
        return reinterpret_cast<T*>(reinterpret_cast<std::ptrdiff_t>(ptr) + reinterpret_cast<std::ptrdiff_t>(other));
    }
    

    void save_model(std::ofstream& file) {
        ggml_context const& ctx = *m_model.ctx;
        save_value(file, ctx.mem_buffer_owned);
        save_value(file, ctx.n_objects);
        save_value(file, get_relative_ptr(ctx.objects_begin));
        save_value(file, get_relative_ptr(ctx.objects_end));
        save_value(file, get_relative_ptr(ctx.scratch.data));
        save_value(file, ctx.scratch.offs);
        save_value(file, ctx.scratch.size);
        save_value(file, get_relative_ptr(ctx.scratch_save.data));
        save_value(file, ctx.scratch_save.offs);
        save_value(file, ctx.scratch_save.size);

        std::size_t map_size = m_model.tensors.size();
        save_value(file, map_size);
    

        for(auto const& [k, v] : m_model.tensors) {
            save_str(file, k);
            save_value(file, get_relative_ptr(v));
        }
        
        transform_tensors_to_relative_ptr(*m_model.tok_embeddings);
        transform_tensors_to_relative_ptr(*m_model.norm);
        transform_tensors_to_relative_ptr(*m_model.output);
        transform_tensors_to_relative_ptr(*m_model.memory_k);
        transform_tensors_to_relative_ptr(*m_model.memory_v);


        for(auto const& l : m_model.layers) {
            transform_tensors_to_relative_ptr(*l.attention_norm);
            transform_tensors_to_relative_ptr(*l.wq);
            transform_tensors_to_relative_ptr(*l.wk);
            transform_tensors_to_relative_ptr(*l.wv);
            transform_tensors_to_relative_ptr(*l.wo);
            transform_tensors_to_relative_ptr(*l.ffn_norm);
            transform_tensors_to_relative_ptr(*l.w1);
            transform_tensors_to_relative_ptr(*l.w2);
            transform_tensors_to_relative_ptr(*l.w3);
        }

        save_value(file, ctx.mem_size);
        // printf("Saved memory Buffer Size: %zu\n\n", ctx.mem_size);
        file.write(reinterpret_cast<char const*>(ctx.mem_buffer), ctx.mem_size);
        file.write(reinterpret_cast<char const*>(&m_model.hparams), sizeof(m_model.hparams));

        auto const save_tensor = [this, &file](ggml_tensor* t) {
            save_value(file, get_relative_ptr(t));
            transform_tensors_to_resolved_ptr(*t);
        };

        save_tensor(m_model.tok_embeddings);
        save_tensor(m_model.norm);
        save_tensor(m_model.output);
        save_tensor(m_model.memory_k);
        save_tensor(m_model.memory_v);

        save_value(file, m_model.layers.size());
        for(auto const& l : m_model.layers) {
            save_tensor(l.attention_norm);
            save_tensor(l.wq);
            save_tensor(l.wk);
            save_tensor(l.wv);
            save_tensor(l.wo);
            save_tensor(l.ffn_norm);
            save_tensor(l.w1);
            save_tensor(l.w2);
            save_tensor(l.w3);
        }
    }

    void transform_tensors_to_relative_ptr(ggml_tensor& t) {
        t.data = get_relative_ptr<void*>(t.data);
        if (t.src0) transform_tensors_to_relative_ptr(*t.src0);
        if (t.src1) transform_tensors_to_relative_ptr(*t.src1);
        if (t.grad) transform_tensors_to_relative_ptr(*t.grad);
        for(auto i = 0ul; i < GGML_MAX_OPT; ++i) {
            if (t.opt[i]) transform_tensors_to_relative_ptr(*t.opt[i]);
        }
    }
    
    void transform_tensors_to_resolved_ptr(ggml_tensor& t) {
        t.data = resolve_relative_ptr<void*>(t.data);
        if (t.src0) transform_tensors_to_resolved_ptr(*t.src0);
        if (t.src1) transform_tensors_to_resolved_ptr(*t.src1);
        if (t.grad) transform_tensors_to_resolved_ptr(*t.grad);
        for(auto i = 0ul; i < GGML_MAX_OPT; ++i) {
            if (t.opt[i]) transform_tensors_to_resolved_ptr(*t.opt[i]);
        }
    }

    template<typename T>
    void load_vec(std::ifstream& file, std::vector<T>& v) const {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        auto value = v.size();
        load_value(file, value);
        v.resize(value);
        file.read(reinterpret_cast<char*>(v.data()), sizeof(T) * value);
    }

    std::string load_str(std::ifstream& file) const {
        std::size_t len = 0;
        load_value(file, len);
        auto s = std::string(' ', len);
        file.read(s.data(), sizeof(char) * len);
        return s;
    }

    void load_vocab(std::ifstream& file) {
        auto len = m_vocab.id_to_token.size();
        load_value(file, len);
        // std::cout<<"Vocab size: "<<len<<'\n';
        for (auto i = 0ul; i < len; ++i) {
            int k = 0;
            load_value(file, k);
            auto v = load_str(file);
            m_vocab.id_to_token[k] = v;
            m_vocab.token_to_id[v] = k;
        }
    }

    void load_model(std::ifstream& file) {
        auto& ctx = *m_model.ctx;

        std::ptrdiff_t val{};
        
        load_value(file, ctx.mem_buffer_owned);
        load_value(file, ctx.n_objects);
        
        load_value(file, val);
        ctx.objects_begin = resolve_relative_ptr<ggml_object>(val);
        load_value(file, val);
        ctx.objects_end = resolve_relative_ptr<ggml_object>(val);

        load_value(file, val);
        ctx.scratch.data = resolve_relative_ptr<void>(val);
        load_value(file, ctx.scratch.offs);
        load_value(file, ctx.scratch.size);

        load_value(file, val);
        ctx.scratch_save.data = resolve_relative_ptr<void>(val);
        load_value(file, ctx.scratch_save.offs);
        load_value(file, ctx.scratch_save.size);

        std::size_t map_size = 0;
        load_value(file, map_size);
        m_model.tensors.clear();

        // printf("Map Size: %zu\n", map_size);
        for(auto i = 0ul; i < map_size; ++i) {
            auto k = load_str(file);
            load_value(file, val);
            // printf("    Map Key: %s\n", k.c_str());
            // printf("    Map Value: %ld\n\n", val);
            auto* ptr = resolve_relative_ptr<ggml_tensor>(val);
            m_model.tensors[std::move(k)] = ptr;
        }

        load_value(file, ctx.mem_size);
        // printf("Memory Buffer Size: %zu\n\n", ctx.mem_size);
        file.read(reinterpret_cast<char*>(ctx.mem_buffer), ctx.mem_size);
        file.read(reinterpret_cast<char*>(&m_model.hparams), sizeof(m_model.hparams));

        auto const load_tensor = [this, &file]() {
            std::ptrdiff_t ptr{};
            load_value(file, ptr);
            auto* t = resolve_relative_ptr<ggml_tensor>(ptr);
            transform_tensors_to_resolved_ptr(*t);
            return t;
        };
        m_model.tok_embeddings = load_tensor();
        m_model.norm = load_tensor();
        m_model.output = load_tensor();
        m_model.memory_k = load_tensor();
        m_model.memory_v = load_tensor();

        m_model.layers.clear();
        auto l_size = m_model.layers.size();
        load_value(file, l_size);
        m_model.layers.reserve(l_size);

        for(auto i = 0ul; i < l_size; ++i) {
            auto l = llama_layer{};
            l.attention_norm = load_tensor();
            l.wq = load_tensor();
            l.wk = load_tensor();
            l.wv = load_tensor();
            l.wo = load_tensor();
            l.ffn_norm = load_tensor();
            l.w1 = load_tensor();
            l.w2 = load_tensor();
            l.w3 = load_tensor();
            m_model.layers.push_back(l);
        }
    }

private:
    llama_model m_model;
    gpt_vocab m_vocab;
    int m_threads;
    int n_past{0};
    int m_seed{0};
    size_t m_mem_per_token = 0;
    std::mt19937 m_rng;
    std::vector<gpt_vocab::id> m_embd;
    std::vector<gpt_vocab::id> m_last_n_tokens;
    std::vector<float> m_logits;
};


PYBIND11_MODULE(fastLlama, m) {
    py::class_<FastLlama>(m, "Model")
        .def(py::init<std::string const&, int, int, std::size_t, int>(), py::arg("path"), py::arg("num_threads"), py::arg("n_ctx"), py::arg("last_n_size") = 200, py::arg("seed") = 0)
        .def("ingest", &FastLlama::ingest, py::arg("prompt"))
        .def("generate", &FastLlama::generate, py::arg("streaming_fn"), py::arg("num_tokens") = 100, py::arg("top_p") = 0.95f, py::arg("temp") = 0.8f, py::arg("repeat_penalty") = 1.f, py::arg("stop_word") = "")
        .def("save_state", &FastLlama::save_state, py::arg("path"))
        .def("load_state", &FastLlama::load_state, py::arg("path"));
}