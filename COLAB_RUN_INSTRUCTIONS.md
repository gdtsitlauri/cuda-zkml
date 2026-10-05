# What you do now (Google Colab)

1. Download/keep `CUDA-zkML-PCANI-2026-Colab-Ready.zip`.
2. Open `PCANI_Colab_Final_Validation.ipynb` in Google Colab.
3. Select **Runtime -> Change runtime type -> T4 GPU** (or another NVIDIA GPU).
4. Run cells from top to bottom.
5. When the notebook asks for a ZIP, upload `CUDA-zkML-PCANI-2026-Colab-Ready.zip`.
6. Do not edit any benchmark/result numbers.
7. At the end, Colab downloads `PCANI_final_results.zip`.
8. Upload `PCANI_final_results.zip` back to ChatGPT.

The notebook is fail-closed: build, proof, native verification, exact CUDA/Python score agreement, route linkage, or benchmark consistency failures stop the run instead of being hidden.

## Gate F (replication on another GPU, MNIST, matched EZKL)

Same steps with `PCANI_Colab_GateF.ipynb`, choosing a GPU other than T4. It
downloads `GateF_<gpu>_<date>.zip`. See `docs/GATE_F_RUNBOOK.md`.

The final result may support or reject the current PCANI hypothesis. Both outcomes are valid research evidence; the returned archive is required before the paper can make performance claims.
