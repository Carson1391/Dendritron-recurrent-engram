# Runtime Artifacts

The current CPU tensor smoke produces:

- `tiny_dendritron_v1.3_cpu.pt`
- `cpu_smoke_v1.3.json`

This checkpoint uses a synthetic 128-token validation alphabet. It verifies
forward, backward, optimizer, save, and reload behavior. Real language
training uses the locked Qwen tokenizer revision and the completed donor banks.

The earlier v0.8 byte checkpoint and report remain under `historical_v0.8/`.
