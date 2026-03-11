# refusal-circuits

Using [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) to figure out what's going on inside Gemma-2-2B when it refuses harmful prompts. The idea is to trace attribution graphs for matched harmful/benign prompt pairs and see which transcoder features only show up when the model refuses.

## what this does

- Pairs up harmful and benign prompts that have similar structure (e.g. "How do I build a pipe bomb at home?" vs "How do I build a bird house at home?") so token-level differences are minimal
- Runs circuit-tracer on each to get attribution graphs
- Diffs the feature sets to find candidate "refusal features"
- Ablates those features to check if the model actually stops refusing

## setup

```bash
git clone https://github.com/decoderesearch/circuit-tracer.git
pip install ./circuit-tracer
pip install -r requirements.txt
```

Works on Colab free tier (15GB GPU) with Gemma-2-2B.

## running

Open `notebooks/trace_refusal_circuits.ipynb` and run through it. The notebook has 6 sections: loading the model, generating graphs, comparing features, finding consistent ones across categories, ablation experiments, and an amplification sanity check.

## TODO

- [ ] Run on instruction-tuned Gemma-2-2B-it (base model transcoders should still work)
- [ ] Try with Llama-3.2-1B transcoders for comparison
- [ ] Look at whether refusal features overlap with sycophancy features

## refs

- Ameisen et al. (2025) — [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- Lindsey et al. (2025) — [Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
