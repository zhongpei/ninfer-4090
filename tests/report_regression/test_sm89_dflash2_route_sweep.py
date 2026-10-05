from tools.bench.sm89_dflash2_k7_routes import parse_sweep

SAMPLE = """# gpu=NVIDIA GeForce RTX 4090 sm=89  q8 dflash2 attn_input (27B), cold, median min..p95 of 31
     T small_t sm89_occ_k7 public_op routed_to winner
     8 10.0 9.0 9.5 attn_input_proj.q8.dflash2.small_t sm89_occ_k7
    16 11.0 10.0 10.4 attn_input_proj.q8.dflash2.small_t sm89_occ_k7
    32 20.0 21.0 20.1 attn_input_proj.q8.dflash2.mma.r32.c32.k128 small_t
    64 40.0 38.0 39.0 attn_input_proj.q8.dflash2.mma.r32.c64.k128 sm89_occ_k7
# gpu=NVIDIA GeForce RTX 4090 sm=89  q8 dflash2 linear_swiglu (27B), cold, median min..p95 of 31
     T small_t sm89_occ_k7 public_op routed_to winner
     8 12.0 11.0 11.5 linear_swiglu.q8.dflash2.small_t sm89_occ_k7
    16 13.0 12.0 12.5 linear_swiglu.q8.dflash2.small_t sm89_occ_k7
    32 18.0 17.0 17.4 linear_swiglu.q8.dflash2.small_t sm89_occ_k7
    64 30.0 29.0 29.2 linear_swiglu.q8.dflash2.mma.r64.c64.k128 sm89_occ_k7
"""


def test_parse_sm89_k7_sweep():
    gpu, sections = parse_sweep(SAMPLE)
    assert gpu == "NVIDIA GeForce RTX 4090"
    assert sections["attn_input"][8]["measured_winner"] == "sm89_occ_k7"
    assert sections["attn_input"][32]["measured_winner"] == "small_t"
    assert sections["linear_swiglu"][64]["measured_winner"] == "sm89_occ_k7"


def test_rejects_non_sm89():
    bad = SAMPLE.replace("sm=89", "sm=86")
    try:
        parse_sweep(bad)
    except RuntimeError as error:
        assert "sm89 qualification" in str(error)
    else:
        raise AssertionError("sm86 output must not qualify sm89 routing")
