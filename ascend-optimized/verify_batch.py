# B>1 correctness check: C++ forward vs Python model, per-sequence cos.
# Result (2026-07-05): B=4 all 4 sequences cos=1.00000, argmax match.
# => batch decode (B=8 1752, B=16 3358 aggregate tok/s) is CORRECT, not just fast.
