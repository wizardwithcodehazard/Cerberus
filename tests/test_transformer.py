"""Test OpenMP and OpenACC Offload Pragma Generation."""

from cerberus.model import ProfitabilityModel
from cerberus.hardware import PRESET_PROFILES
from cerberus.transformer import GPUPragmaTransformer

def test_openmp_and_openacc_transformation():
    source = """
    void matmul(float* A, float* B, float* C) {
        for (int i = 0; i < 512; i++) {
            for (int k = 0; k < 512; k++) {
                for (int j = 0; j < 512; j++) {
                    C[i * 512 + j] += A[i * 512 + k] * B[k * 512 + j];
                }
            }
        }
    }
    """
    model = ProfitabilityModel()
    hw = PRESET_PROFILES["dgpu_rtx3060"]

    # 1. Test OpenMP
    omp_transformer = GPUPragmaTransformer(model, hw, dialect="openmp")
    omp_code, omp_decisions = omp_transformer.transform_source(source, speedup_threshold=1.0)
    assert len(omp_decisions) == 1
    assert omp_decisions[0][2] is True # Profitable
    assert "#pragma omp target teams distribute parallel for" in omp_code
    assert "map(to: A" in omp_code or "map(to: B" in omp_code
    assert "map(tofrom: C" in omp_code

    # 2. Test OpenACC
    acc_transformer = GPUPragmaTransformer(model, hw, dialect="openacc")
    acc_code, acc_decisions = acc_transformer.transform_source(source, speedup_threshold=1.0)
    assert len(acc_decisions) == 1
    assert acc_decisions[0][2] is True # Profitable
    assert "#pragma acc parallel loop" in acc_code
    assert "copyin(" in acc_code
    assert "copy(C" in acc_code

def test_dynamic_if_crossover_generation():
    dynamic_source = """
    void dynamic_matmul(float* A, float* B, float* C, int N) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                float aik = A[i * N + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += aik * B[k * N + j];
                }
            }
        }
    }
    """
    model = ProfitabilityModel()
    hw = PRESET_PROFILES["dgpu_rtx3060"]

    # OpenMP with dynamic if(N >= ...) clause
    omp_transformer = GPUPragmaTransformer(model, hw, dialect="openmp")
    omp_code, decisions = omp_transformer.transform_source(dynamic_source, speedup_threshold=1.0)
    assert len(decisions) == 1
    assert decisions[0][2] is True
    assert "if(N >=" in omp_code
    assert "map(to: A[0:N], B[0:N])" in omp_code or "map(to: B[0:N], A[0:N])" in omp_code
    assert "map(tofrom: C[0:N])" in omp_code

    # OpenACC with dynamic if(N >= ...) clause
    acc_transformer = GPUPragmaTransformer(model, hw, dialect="openacc")
    acc_code, _ = acc_transformer.transform_source(dynamic_source, speedup_threshold=1.0)
    assert "if(N >=" in acc_code
    assert "copyin(" in acc_code

if __name__ == "__main__":
    test_openmp_and_openacc_transformation()
    test_dynamic_if_crossover_generation()
    print("All transformer tests passed!")
