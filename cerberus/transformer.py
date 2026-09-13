"""Multi-Dialect GPU Offload Pragma Transformer (OpenMP 4.5+ & OpenACC)."""

import re
import copy
from typing import List, Tuple, Optional
from cerberus.parser import LoopFeature, CLoopParser
from cerberus.model import PredictionResult, ProfitabilityModel
from cerberus.hardware import HardwareProfile

class GPUPragmaTransformer:
    """Source-to-source rewriter that injects OpenMP 4.5+ or OpenACC offload pragmas on profitable loops."""

    def __init__(self, model: ProfitabilityModel, target_hw: HardwareProfile, dialect: str = "openmp"):
        self.model = model
        self.target_hw = target_hw
        self.dialect = dialect.lower() # "openmp" or "openacc"

    def transform_source(self, source_code: str, speedup_threshold: float = 1.1) -> Tuple[str, List[Tuple[LoopFeature, PredictionResult, bool]]]:
        """Analyzes all loops in source_code and injects pragmas only for profitable ones."""
        parser = CLoopParser()
        loops = parser.parse_source(source_code)

        lines = source_code.splitlines()
        decisions = []

        # Process loops in reverse line order so line offset shifts don't affect previous loops
        for loop in reversed(loops):
            pred = self.model.predict_loop(loop, self.target_hw, speedup_threshold)
            
            should_offload = pred.is_profitable and loop.is_parallel_safe
            decisions.append((loop, pred, should_offload))

            if should_offload:
                pragma_str = self._generate_pragma(loop)
                # Insert pragma above loop.line_start (0-indexed: loop.line_start - 1)
                insert_idx = loop.line_start - 1
                lines.insert(insert_idx, pragma_str)

        transformed_code = "\n".join(lines)
        decisions.reverse() # Restore original top-to-bottom order
        return transformed_code, decisions

    def find_crossover_threshold(self, loop: LoopFeature) -> Optional[int]:
        """Calculates the minimum problem size N where GPU offloading becomes profitable (Speedup >= 1.05x)."""
        test_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]
        
        for sz in test_sizes:
            scaled = copy.copy(loop)
            if loop.nesting_depth >= 3:
                effective_trips = sz * sz * sz
                unique_mem = 3 * (sz * sz)
                scaled.data_reuse_ratio = float(sz)
            elif loop.nesting_depth == 2:
                effective_trips = sz * sz
                unique_mem = max(1, len(loop.arrays_read | loop.arrays_written)) * (sz * sz)
                scaled.data_reuse_ratio = 2.0
            else:
                effective_trips = sz
                unique_mem = max(1, len(loop.arrays_read | loop.arrays_written)) * sz
                scaled.data_reuse_ratio = 1.0

            scaled.trip_count = effective_trips
            scaled.total_flops = max(1, effective_trips * scaled.flops_per_iter)
            scaled.memory_footprint_bytes = max(1024, unique_mem * 4) # 4 bytes per float
            scaled.arithmetic_intensity = scaled.total_flops / float(scaled.memory_footprint_bytes)

            pred = self.model.predict_loop(scaled, self.target_hw, speedup_threshold=1.05)
            if pred.is_profitable:
                return sz

        return None

    def _get_dynamic_bound_var(self, loop: LoopFeature) -> Optional[str]:
        """Extracts dynamic variable bound from loop header (e.g. 'N' in 'for (int i = 0; i < N; ++i)')."""
        bound_match = re.search(r'for\s*\([^;]*;\s*\w+\s*<\s*([a-zA-Z_]\w*)\s*;', loop.source_code)
        if bound_match:
            var_name = bound_match.group(1)
            # If it's a variable identifier (not a pure number), it is a dynamic runtime bound
            if not var_name.isdigit():
                return var_name
        return None

    def _generate_pragma(self, loop: LoopFeature) -> str:
        """Generates appropriate OpenMP or OpenACC target offload directive."""
        if self.dialect == "openacc":
            return self._generate_openacc_pragma(loop)
        return self._generate_openmp_pragma(loop)

    def _generate_openmp_pragma(self, loop: LoopFeature) -> str:
        """Generates OpenMP 4.5+ target offload directive with if() and array section annotations."""
        clauses = ["#pragma omp target teams distribute parallel for"]

        # 1. Dynamic runtime crossover if() clause
        bound_var = self._get_dynamic_bound_var(loop)
        if bound_var:
            crossover_n = self.find_crossover_threshold(loop)
            if crossover_n:
                clauses.append(f"if({bound_var} >= {crossover_n})")

        # 2. Reduction clause if applicable
        if loop.has_reduction and loop.reduction_var:
            clauses.append(f"reduction(+:{loop.reduction_var})")

        # 3. Map clauses for arrays with size annotations
        read_only = loop.arrays_read - loop.arrays_written
        write_only = loop.arrays_written - loop.arrays_read
        read_write = loop.arrays_read & loop.arrays_written

        size_hint = self._get_array_size_hint(loop)

        if read_only:
            arr_list = ", ".join(self._annotate_arrays(sorted(read_only), size_hint))
            clauses.append(f"map(to: {arr_list})")
        if write_only:
            arr_list = ", ".join(self._annotate_arrays(sorted(write_only), size_hint))
            clauses.append(f"map(from: {arr_list})")
        if read_write:
            arr_list = ", ".join(self._annotate_arrays(sorted(read_write), size_hint))
            clauses.append(f"map(tofrom: {arr_list})")

        return " ".join(clauses)

    def _generate_openacc_pragma(self, loop: LoopFeature) -> str:
        """Generates OpenACC 2.7+ parallel loop directive with if() and data clauses."""
        clauses = ["#pragma acc parallel loop"]

        # 1. Dynamic runtime crossover if() clause
        bound_var = self._get_dynamic_bound_var(loop)
        if bound_var:
            crossover_n = self.find_crossover_threshold(loop)
            if crossover_n:
                clauses.append(f"if({bound_var} >= {crossover_n})")

        # 2. Reduction clause if applicable
        if loop.has_reduction and loop.reduction_var:
            clauses.append(f"reduction(+:{loop.reduction_var})")

        # 3. Copy clauses for arrays with size annotations
        read_only = loop.arrays_read - loop.arrays_written
        write_only = loop.arrays_written - loop.arrays_read
        read_write = loop.arrays_read & loop.arrays_written

        size_hint = self._get_array_size_hint(loop)

        if read_only:
            arr_list = ", ".join(self._annotate_arrays(sorted(read_only), size_hint))
            clauses.append(f"copyin({arr_list})")
        if write_only:
            arr_list = ", ".join(self._annotate_arrays(sorted(write_only), size_hint))
            clauses.append(f"copyout({arr_list})")
        if read_write:
            arr_list = ", ".join(self._annotate_arrays(sorted(read_write), size_hint))
            clauses.append(f"copy({arr_list})")

        return " ".join(clauses)

    def _get_array_size_hint(self, loop: LoopFeature) -> Optional[str]:
        """Extracts the best available array size from the loop's source context."""
        bound_match = re.search(r'for\s*\([^;]*;\s*\w+\s*<\s*([a-zA-Z_]\w*)\s*;', loop.source_code)
        if bound_match:
            return bound_match.group(1)
        return None

    def _annotate_arrays(self, arr_names: list, size_hint: Optional[str]) -> list:
        """Adds [0:N] section annotation to array names if size hint is available."""
        if size_hint:
            return [f"{a}[0:{size_hint}]" for a in arr_names]
        return arr_names

# Backward-compatibility alias
OpenMPTransformer = GPUPragmaTransformer

