# Explicit remote-base experiment

The v2 communication-loop experiment separated four FP16x8 loads, the
eight-element FP32 reduction, and four FP16x8 stores.  It also attempted to
bind eight peer bases explicitly with:

```python
ptr = T.LetStmt(T.remote_address(T.region(base[...], access, ...), peer), T.ptr("float16"))
flat = T.decl_buffer(shape, "float16", data=ptr, scope="global")
```

The typed pointer construction and `decl_buffer` both parsed, but lowering
currently fails in `normalize_remote_views`: its generic Python mutator does
not rebuild an already-explicit `tl.remote_address` call safely.  This is an
existing frontend compatibility gap, not a reason to change a backend pass in
this prototype.  The operator therefore uses typed `remote_view` aliases
again while retaining the split vector-transfer schedule.  Revisit explicit
bases only with a narrowly reviewed normalizer compatibility fix.

Captured lowering logs are retained under `build-old/gemm-ar-20261011/` as
`source2048-v2*.log`.
