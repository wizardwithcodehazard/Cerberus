"""Test OpenMP and OpenACC Offload Pragma Generation."""

from cerberus.model import ProfitabilityModel
from cerberus.hardware import PRESET_PROFILES
from cerberus.transformer import GPUPragmaTransformer

def test_openmp_and_openacc_transformation():
    source = """
    void saxpy(float a, float* x, float* y, int n) {
        for (int i = 0; i < 1000000; i++) {
            y[i] = a * x[i] + y[i];
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
    assert "map(to: x)" in omp_code
    assert "map(tofrom: y)" in omp_code

    # 2. Test OpenACC
    acc_transformer = GPUPragmaTransformer(model, hw, dialect="openacc")
    acc_code, acc_decisions = acc_transformer.transform_source(source, speedup_threshold=1.0)
    assert len(acc_decisions) == 1
    assert acc_decisions[0][2] is True # Profitable
    assert "#pragma acc parallel loop" in acc_code
    assert "copyin(x)" in acc_code
    assert "copy(y)" in acc_code

if __name__ == "__main__":
    test_openmp_and_openacc_transformation()
    print("All transformer tests passed!")
