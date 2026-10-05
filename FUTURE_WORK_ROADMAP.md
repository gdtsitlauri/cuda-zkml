# CUDA-zkML / PCANI — Roadmap για το μέλλον (PhD)

Κατάσταση: 2026-10-03. Το project «παγώνει» εδώ και συνεχίζεται στο διδακτορικό.
Αυτό το αρχείο λέει **τι μένει, με ποια σειρά, πού στον κώδικα και πώς ελέγχεται**.

## Ενημέρωση 2026-10-04: η Φάση 1 και η Φάση 2 υλοποιήθηκαν, μένουν μόνο τρεξίματα

Ο κώδικας για κάθε επιμέρους στόχο της Φάσης 1 και της Φάσης 2 είναι γραμμένος.
Το τμήμα που δεν χρειάζεται GPU (κυκλώματα, witness, Poseidon, τελετή, Python)
ελέγχθηκε τοπικά. Η πλήρης GPU αλυσίδα (setup, proof, verify) τρέχει με **ένα
notebook**: `PCANI_Colab_Phase12.ipynb`, που καλεί το `scripts/run_phase12.sh`.

| Στόχος | Υλοποίηση | Τοπικός έλεγχος (χωρίς GPU) | Μένει |
|---|---|---|---|
| 1.1 context / anti-replay | `--statement-v2 --context`, `ctx·ctx=ctx_sq` (`src/prover/gadgets.cuh`), έλεγχος `expected_context` / `seen_contexts` (`python/zkml/pcani_protocol_v2.py`), `contracts/PCANIContextVerifier.sol` | ✅ host + pytest· διορθώθηκε **input aliasing** στο `Verifier.sol` (έλεγχος < q αντί < r) | τρέξιμο E1 |
| 1.2 underconstrained | export σε `.r1cs`/`.wtns` (iden3, `src/prover/r1cs_export.cuh`), perturbation check σε C++, ανεξάρτητος έλεγχος `tests/test_underconstrained.py` | ✅ 0 ελεύθερες μεταβλητές (8.082 και 62.692 constraints) | E6· προαιρετικά Picus/SMT |
| 1.3 threat model | `docs/THREAT_MODEL.md` | ✅ | — |
| 2.1 ReLU με range proofs | `RELU_EXACT` (bits + rescale), `python/export_trained_relu_model.py` | ✅ εκπαιδευμένο digits: float 97.3% → integer 97.1%, C++ = Python bit-προς-bit | E1 |
| 2.2 native conv | `CONV2D` (αραιές γραμμές, μόνο μη-μηδενικά taps) | ✅ | E2 |
| 2.3 ιδιωτικό μοντέλο | `--private-model`, Poseidon BN254 (**ίδιο με circomlib**: διάνυσμα αναφοράς και `poseidon([1,2])`) | ✅ | E4 |
| 2.4 ιδιωτική είσοδος | `--private-input --input-blinding`, routing μέσα στο κύκλωμα (top-1/top-2, flag), `--private-outputs` | ✅ | E4 |
| 2.5 τελετή MPC | `zkml-ceremony contribute/verify` (`src/prover/ceremony.cuh`), PoK + pairings | ✅ ακριβής άλγεβρα + απόρριψη αλλοιώσεων (9/9) | E5· πραγματικοί συμμετέχοντες (Φάση 3)· το phase 1 παραμένει single-party |
| 2.6 batching | `--batch-inputs` | ✅ | E3 (χρόνοι) |

Τρέξιμο τώρα: `PCANI_Colab_Phase12.ipynb` (οποιαδήποτε NVIDIA GPU) → `Phase12_<gpu>_<date>.zip`.
Τοπικά, χωρίς CUDA: `tests/host/run_host_tests.sh` και `pytest`.
Η διαδρομή v1 (τα δημοσιευμένα αποτελέσματα T4) **δεν άλλαξε** και αναπαράγεται όπως πριν.

Βασική αρχή: η ασφάλεια πρέπει να στηρίζεται μόνο στα κλειδιά και στις
κρυπτογραφικές υποθέσεις, όχι στη μυστικότητα του κώδικα (Kerckhoffs).
Στόχος δεν είναι «μηδέν αδυναμίες», αλλά ρητό threat model + άμυνα για ό,τι
είναι μέσα σε αυτό + καθαρή δήλωση για ό,τι μένει εκτός.

---

## Φάση 0 — Μόνο τρεξίματα (όλα έτοιμα)

Τίποτα δεν χρειάζεται κώδικα. Λεπτομέρειες: `docs/GATE_F_RUNBOOK.md`.

| Βήμα | Πώς |
|---|---|
| Επανάληψη protocol v2 σε 2η GPU (όχι T4) | `PCANI_Colab_GateF.ipynb` (L4/A100) ή `bash scripts/run_gate_f.sh` |
| MNIST-MLP και MNIST-CNN | ίδιο script (`WORKLOADS="mnist_mlp mnist_conv"`) |
| Δίκαιη σύγκριση με EZKL | ίδιο script (`EZKL=1`) |

Αποτέλεσμα: `GateF_<gpu>_<date>.zip` → νούμερα για το πρώτο paper.

Πιθανά θέματα: μνήμη σε κάρτα 4 GB για το `mnist_conv p12`· αν οι χρόνοι των
paths δεν βγουν αύξοντες, αύξησε `REPEATS`.

---

## Φάση 1 — Πριν από το πρώτο paper (μικρές αλλαγές)

### 1.1 Δέσμευση του proof σε context (anti-replay)
- **Πρόβλημα:** τα Groth16 proofs είναι malleable και ένα έγκυρο proof μπορεί
  να ξαναχρησιμοποιηθεί σε άλλο αίτημα.
- **Λύση:** επιπλέον δημόσια είσοδος `context = H(session_id ‖ nonce ‖ timestamp ‖ policy_digest)`
  δεμένη με μια τετριμμένη constraint στο κύκλωμα, όπως το `model_tag`.
- **Αρχεία:** `src/cli/prove.cu` (`build_fixed_model_circuit`, statement meta),
  `src/prover/witness.cuh`, `python/zkml/pcani_protocol_v2.py` (έλεγχος context
  στο certificate), `contracts/Verifier.sol` (+1 public input).
- **Έλεγχος:** test όπου proof με λάθος context απορρίπτεται.

### 1.2 Έλεγχος για underconstrained κύκλωμα
- **Πρόβλημα:** η πιο συχνή κατηγορία πραγματικών bugs στο zkML. Το κύκλωμα
  μπορεί να δέχεται witness που δεν αντιστοιχεί στο σωστό inference.
- **Λύση:** (α) εξαγωγή του R1CS σε αρχείο (π.χ. μορφή `.r1cs` του circom),
  (β) αρνητικά tests: αλλοιωμένο witness πρέπει να αποτυγχάνει στο
  `R1CS::verify`, (γ) έλεγχος ότι κάθε μεταβλητή εξόδου καθορίζεται μοναδικά
  (εργαλεία τύπου Picus / Circomspect ή δικό μας SMT check σε μικρά κυκλώματα).
- **Αρχεία:** `src/prover/circuit.cuh`, νέο `tests/test_underconstrained.py`.

### 1.3 Ρητό threat model στο paper
- Το μοντέλο είναι **δημόσιο** (σταθερές μέσα στο κύκλωμα → εξάγεται από pk/vk).
- Η είσοδος είναι **δημόσια**· η δρομολόγηση είναι **δημόσια**.
- Το trusted setup ανά μοντέλο είναι **υπόθεση εμπιστοσύνης**.

---

## Φάση 2 — Επεκτάσεις διδακτορικού

### 2.1 Πραγματικό ReLU με range proofs (κύριο τεχνικό κενό)
- **Σήμερα:** ενεργοποίηση `h = z² + 64z + 64` και τυχαία, μη εκπαιδευμένα
  κρυφά βάρη (`python/pcani_prepare.py`).
- **Λύση:** ανάλυση του z σε bits μέσα στο κύκλωμα (k constraints για k-bit
  εύρος) → bit προσήμου s → `relu(z) = (1 - s)·z`. Μετά, εκπαιδευμένα πολυεπίπεδα
  μοντέλα με quantization (int8/int16) και rescaling μέσω range-checked διαίρεσης.
- **Αρχεία:** νέος `LayerType::RELU_EXACT` σε `src/nn/layers.cuh`,
  `inference.cu`, `circuit.cuh` (`add_relu_exact`), `witness.cuh`·
  `python/zkml/model.py` και νέος exporter εκπαιδευμένου μοντέλου (PyTorch).
- **Έλεγχος:** ακριβής ισότητα CUDA/Python logits (όπως σήμερα) σε εκπαιδευμένο
  MNIST-CNN, αρνητικό test με λάθος bit.

### 2.2 Native convolution layer
- **Σήμερα:** το CNN ξεδιπλώνεται σε πυκνό πίνακα με πολλά μηδενικά.
- **Λύση:** `LayerType::CONV2D` με αραιές γραμμικές constraints (μόνο τα μη
  μηδενικά βάρη). Παράλληλα, παράλειψη μηδενικών όρων στο
  `add_fixed_linear_layer_into` (προσοχή: αλλάζει το R1CS του Digits, άρα
  ξεχωριστό πείραμα και όχι αντικατάσταση του T4 αποτελέσματος).

### 2.3 Ιδιωτικό μοντέλο (προστασία από εξαγωγή βαρών / reverse engineering)
- **Σήμερα:** όποιος έχει pk/vk μπορεί να διαβάσει όλα τα βάρη.
- **Λύση:** βάρη ως ιδιωτικό witness + δημόσια δέσμευση
  `C_model = Poseidon(weights)` υπολογισμένη μέσα στο κύκλωμα.
  Χρειάζεται κύκλωμα Poseidon (BN254) σε R1CS και αντίστοιχος witness generator.
  Ένα universal pk/vk ανά **αρχιτεκτονική**, όχι ανά μοντέλο.
- **Κόστος:** οι πολλαπλασιασμοί βάρος×είσοδος γίνονται πραγματικές constraints
  (σήμερα είναι «δωρεάν» γραμμικές) → πολύ μεγαλύτερο κύκλωμα. Είναι
  ερευνητικό ερώτημα από μόνο του (κόστος ιδιωτικότητας vs PCANI savings).

### 2.4 Ιδιωτική είσοδος
- Ίδιος μηχανισμός: `C_input = Poseidon(x ‖ r)` δημόσιο, x ιδιωτικό.
- Η δρομολόγηση (επιλογή path) πρέπει τότε να μπει **μέσα** στο κύκλωμα
  (σύγκριση margin με threshold μέσω range proof), αλλιώς διαρρέει πληροφορία.

### 2.5 Trusted setup
- **Σήμερα:** ένα setup ανά μοντέλο· όποιος το τρέχει μπορεί να πλαστογραφεί.
- **Ρεαλιστική λύση:** εργαλείο τελετής MPC phase-2 για Groth16 (κάθε
  συμμετέχων πολλαπλασιάζει το δ με τυχαίο s και ενημερώνει τα L/H queries,
  με απόδειξη συνέπειας). Αρχεία: `src/prover/groth16.cu` (ProvingKey),
  νέο CLI `zkml-ceremony`.
- **Μεγάλη εναλλακτική:** universal (PLONK/KZG) ή transparent (STARK/Halo2)
  setup → ουσιαστικά νέος prover· μόνο αν γίνει κεντρικό θέμα του PhD.

### 2.6 Κλίμακα
- Batching πολλών δειγμάτων σε ένα proof· μεγαλύτερα μοντέλα (μικρό ResNet,
  μικρός transformer — υπάρχει ήδη `SELF_ATTENTION` στο κύκλωμα).

---

## Φάση 3 — Δεν λύνεται με κώδικα

| Θέμα | Τι χρειάζεται |
|---|---|
| Systematic literature/patent novelty audit | διάβασμα — λέξεις-κλειδιά στο `docs/NOVELTY_GATE_2026.md` |
| Εξωτερικός κρυπτογραφικός έλεγχος | κρυπτογράφος (επιβλέπων / συνεργάτης) |
| Τελετή MPC | πραγματικοί συμμετέχοντες |
| Side channels του GPU prover (αν γίνει ιδιωτικό witness) | μετρήσεις χρόνου/μνήμης· να δηλωθεί στο threat model |

---

## Εργαλεία που χρειάζονται για τη Φάση 1–2

- Linux ή **WSL2 + Ubuntu** με CUDA toolkit (η GTX 1650 / sm_75 αρκεί για
  ανάπτυξη)· Colab για τελικά τρεξίματα.
- Python: `numpy scikit-learn pytest onnx ezkl` (+ `torch` για εκπαιδευμένα μοντέλα).

## Προτεινόμενη σειρά

1. Φάση 0 (τρεξίματα) → πρώτο paper.
2. 1.1 + 1.2 + 1.3.
3. 2.1 (ReLU) → 2.2 → δεύτερο paper (εκπαιδευμένα μοντέλα).
4. 2.3 + 2.4 + 2.5 → τρίτο paper (ιδιωτικό PCANI).
