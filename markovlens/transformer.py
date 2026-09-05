"""OpenMP 4.5+ Target Offload Pragma Transformer & Gating Injector."""

import re
from typing import List, Tuple
from markovlens.parser import LoopFeature
from markovlens.model import PredictionResult, ProfitabilityModel
from markovlens.hardware import HardwareProfile

class OpenMPTransformer:
    """Source-to-source rewriter that injects OpenMP target offload pragmas on profitable loops."""

    def __init__(self, model: ProfitabilityModel, target_hw: HardwareProfile):
        self.model = model
        self.target_hw = target_hw

    def transform_source(self, source_code: str, speedup_threshold: float = 1.1) -> Tuple[str, List[Tuple[LoopFeature, PredictionResult, bool]]]:
        """Analyzes all loops in source_code and injects OpenMP pragmas only for profitable ones."""
        from markovlens.parser import CLoopParser
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

    def _generate_pragma(self, loop: LoopFeature) -> str:
        """Generates appropriate OpenMP 4.5+ target offload directive."""
        clauses = ["#pragma omp target teams distribute parallel for"]

        # 1. Reduction clause if applicable
        if loop.has_reduction and loop.reduction_var:
            clauses.append(f"reduction(+:{loop.reduction_var})")

        # 2. Map clauses for arrays
        read_only = loop.arrays_read - loop.arrays_written
        write_only = loop.arrays_written - loop.arrays_read
        read_write = loop.arrays_read & loop.arrays_written

        if read_only:
            clauses.append(f"map(to: {', '.join(sorted(read_only))})")
        if write_only:
            clauses.append(f"map(from: {', '.join(sorted(write_only))})")
        if read_write:
            clauses.append(f"map(tofrom: {', '.join(sorted(read_write))})")

        return " ".join(clauses)
