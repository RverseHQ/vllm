// The upstream PACKGQA_ONLY dispatcher, restricted to the separately loaded
// SM90 FP8 D256 library. All runtime scheduling and attention parameters still
// come from the unmodified b273 mha_fwd implementation.
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    STD_TORCH_CHECK(params.arch == 90 && params.is_e4m3
                    && params.d == 256 && params.dv == 256,
                    "The isolated FA3 library supports only SM90 FP8 D256");
    STD_TORCH_CHECK(params.num_splits >= 1);
    SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
        PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256,
                             Split, PagedKVNonTMA, Has_softcap, true>(params, stream);
            });
        });
    });
}
