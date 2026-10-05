#include "prover/witness.cuh"
#include "prover/circuit.cuh"
#include <cstdio>
using namespace zkml;
int main() {
  CircuitBuilder b;
  int x = b.alloc_var(), y = b.alloc_var(), z = b.alloc_var();
  b.circuit.add_mul_constraint(x, Fr::one(), y, Fr::one(), z, Fr::one());
  b.finalize(1, 2);
  std::vector<Fr> w = {Fr::one(), Fr::from_uint(3), Fr::from_uint(5), Fr::from_uint(15)};
  bool ok = b.circuit.verify_witness(w);
  w[3] = Fr::from_uint(16);
  bool bad = b.circuit.verify_witness(w);
  printf("ok=%d bad=%d\n", ok, bad);
  return (ok && !bad) ? 0 : 1;
}
