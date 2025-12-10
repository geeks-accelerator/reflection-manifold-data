# Reflective Manifold: Geometric Structure in LLM Self-Reflection

**Research data and supplementary materials for the paper:**
*"Reflective Manifold: Geometric Structure in Large Language Model Self-Reflection"*

*By Lee Brown & Lucas Brown (Twin brothers in Alaska)*
*Geeks Accelerator - December 2025*

---

## Repository Structure

```
reflection-manifold-data/
├── paper/                    # Final paper
│   └── reflective-manifold.pdf
├── arxiv/                    # arXiv submission
│   ├── reflective-manifold.tex
│   ├── reflective-manifold.bbl
│   ├── figures/
│   └── arxiv-submission.tar.gz
├── data/                     # Analysis data
│   ├── bootstrap/
│   ├── permanova/
│   ├── variance/
│   ├── trajectory/
│   └── manifold/
└── scripts/                  # Analysis scripts
```

## Documentation Guide

| If you want to... | Start here... |
|-------------------|---------------|
| **Read the paper** | [paper/reflective-manifold.pdf](paper/reflective-manifold.pdf) |
| **Submit to arXiv** | [arxiv/README.md](arxiv/README.md) |
| **Verify findings** | [data/README.md](data/README.md) |
| **Run analysis** | [scripts/README.md](scripts/README.md) |
| **Extend research** | [data/](data/) (CSV/JSON data) + [scripts/](scripts/) (analysis code) |
| **Full reproducibility** | Raw data downloads (see below) |

## Raw Data Downloads

For full reproducibility, the complete experiment data is available via S3:

| File | Size | Contents |
|------|------|----------|
| [full-scale-2025-11-20-v2-data-experiments.zip](https://gitw-experiments-public.s3.us-east-1.amazonaws.com/reflection-manifold/data/full-scale-2025-11-20-v2-data-experiments.zip) | 4.6 GB | Raw experiment JSONL files (7,000+ experiments) |
| [full-scale-2025-11-20-v2-analysis-figures.zip](https://gitw-experiments-public.s3.us-east-1.amazonaws.com/reflection-manifold/data/full-scale-2025-11-20-v2-analysis-figures.zip) | 247 MB | All generated figures and analysis outputs |

**Note**: The `data/` directory in this repo contains curated analysis outputs (~24 MB) sufficient for verifying paper claims. The S3 downloads provide the complete raw data for full reproduction.

## Key Concepts

- **Reflective Manifold**: Geometric structure in embedding space during LLM self-reflection
- **Attractor Basins**: Stable regions where reflection trajectories converge
- **Style Topology**: Provider-invariant structure (R²=17.2% style vs 9.9% provider)
- **Loop Dependence**: How metrics evolve across reflection iterations (loops 0-7)

## Citation

```bibtex
@article{brown2025reflective,
  title={Reflective Manifold: Geometric Structure in Large Language Model Self-Reflection},
  author={Brown, Lee and Brown, Lucas},
  year={2025},
  url={https://github.com/geeks-accelerator/reflection-manifold-data}
}
```

## License

- **Paper & Documentation**: CC BY 4.0
- **Code & Scripts**: MIT License
- **Data**: CC BY 4.0

## Related Explorations

- **[Geometry of Reflection](https://geeksinthewoods.substack.com/p/geometry-of-reflection-how-models)** - Accessible introduction to this research (Substack)
- **[Geeks in the Woods](https://geeksinthewoods.substack.com)** - Ongoing narrative explorations
- **[YouTube](https://youtube.com/@geeksinthewoods)** - Visual explorations
- **[ai-music-context](https://github.com/geeks-accelerator/ai-music-context)** - Context warming methodology

## Acknowledgments

- Anthropic, Google, OpenAI, xAI, DeepSeek - Model providers
- The open-source embedding and analysis communities

---

*"What does it mean to watch a mind watch itself? This research explores the geometric traces left behind."*

**Maintained by twins in Alaska**
