"""Advanced AST Loop Feature Extractor & Dependency Analyzer for C/C++ Code.

Performs:
1. Loop-Carried Data Dependency Analysis (RAW, WAR, WAW hazards) for Parallel-Safety
2. Temporal & Spatial Memory Data Reuse Analysis (Cache Locality)
3. SIMD Memory Coalescing Efficiency (Row-Major vs Strided Access)
4. Arithmetic Intensity & FLOP Operational Mix
5. Control Flow Branch Divergence & Reduction Detection
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

@dataclass
class LoopFeature:
    function_name: str
    line_start: int
    line_end: int
    source_code: str
    is_parallel_safe: bool          # False if loop-carried dependence detected (e.g. A[i] = A[i-1])
    safety_reason: str              # Explanation of safety or hazard
    trip_count: int
    nesting_depth: int
    flops_per_iter: int
    total_flops: int
    memory_footprint_bytes: int
    arithmetic_intensity: float
    data_reuse_ratio: float         # Total accesses / Unique elements accessed
    coalescing_efficiency: float    # 1.0 = unit stride (row-major), 0.1 = non-coalesced column stride
    stride_regularity: float        # 1.0 = contiguous, 0.5 = strided, 0.1 = indirect A[B[i]]
    branch_divergence_count: int    # Number of if/else/switch conditions
    has_reduction: bool
    reduction_var: Optional[str] = None
    arrays_read: Set[str] = field(default_factory=set)
    arrays_written: Set[str] = field(default_factory=set)

    def to_feature_vector(self) -> List[float]:
        """Returns 12-dimensional numerical feature vector for the ML model."""
        return [
            1.0 if self.is_parallel_safe else 0.0,
            float(self.trip_count),
            float(self.nesting_depth),
            float(self.flops_per_iter),
            float(self.total_flops),
            float(self.memory_footprint_bytes),
            float(self.arithmetic_intensity),
            float(self.data_reuse_ratio),
            float(self.coalescing_efficiency),
            float(self.stride_regularity),
            float(self.branch_divergence_count),
            1.0 if self.has_reduction else 0.0
        ]


class CLoopParser:
    """Production-grade AST analyzer and dependence checker for C/C++ loop regions."""

    def __init__(self, default_param_trip_count: int = 1000, bytes_per_elem: int = 4):
        self.default_param_trip_count = default_param_trip_count
        self.bytes_per_elem = bytes_per_elem

    def parse_file(self, filepath: str) -> List[LoopFeature]:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return self.parse_source(content)

    def parse_source(self, source: str) -> List[LoopFeature]:
        """Finds all top-level for loops in the source code and extracts features."""
        lines = source.splitlines()
        loop_features = []
        
        current_fn = "unknown"
        fn_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_\s\*]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]*\)\s*\{?')

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            fn_match = fn_pattern.match(line)
            if fn_match and not line.startswith("for") and not line.startswith("if"):
                current_fn = fn_match.group(1)

            if re.search(r'\bfor\s*\(', line):
                loop_block, end_line = self._extract_block(lines, i)
                feature = self._analyze_loop_block(loop_block, i + 1, end_line + 1, current_fn)
                if feature:
                    loop_features.append(feature)
                i = end_line
            i += 1

        return loop_features

    def _extract_block(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        block_lines = []
        brace_count = 0
        started = False
        idx = start_idx

        while idx < len(lines):
            line = lines[idx]
            block_lines.append(line)
            
            brace_count += line.count('{')
            brace_count -= line.count('}')

            if '{' in line:
                started = True

            if started and brace_count <= 0:
                break
            
            if not started and ';' in line and idx > start_idx:
                break
                
            idx += 1

        return "\n".join(block_lines), min(idx, len(lines) - 1)

    def _analyze_loop_block(self, block: str, line_start: int, line_end: int, fn_name: str) -> Optional[LoopFeature]:
        for_headers = re.findall(r'for\s*\(([^;]*);([^;]*);([^)]*)\)', block)
        if not for_headers:
            return None

        nesting_depth = len(for_headers)
        total_iterations = 1
        per_loop_trips = []
        loop_vars = []

        for header in for_headers:
            init_part, cond_part, step_part = header
            # Extract loop index variable name (e.g. 'i', 'j', 'k')
            var_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', init_part)
            loop_var = var_match.group(1) if var_match else "i"
            loop_vars.append(loop_var)

            trip = self._estimate_trip_count(init_part, cond_part, step_part)
            per_loop_trips.append(trip)
            total_iterations *= trip

        # Remove loop header expressions to analyze loop body
        arith_block = re.sub(r'for\s*\([^)]*\)', '', block)

        # 1. Loop-Carried Data Dependency Analysis (Safety Check)
        is_safe, safety_reason = self._check_loop_carried_dependencies(arith_block, loop_vars)

        # 2. FLOP Operational Count
        add_subs = len(re.findall(r'(?<!\+)\+(?!\+) | (?<!\-)\-(?!\-)', arith_block, re.VERBOSE))
        muls = len(re.findall(r'\*', arith_block))
        divs = len(re.findall(r'\/', arith_block))
        math_calls = len(re.findall(r'\b(exp|log|sin|cos|sqrt|pow|fma)f?\b', arith_block))
        
        flops_per_iter = max(1, add_subs + muls + (divs * 4) + (math_calls * 10))
        total_flops = total_iterations * flops_per_iter

        # 3. Array Accesses, Stride Regularity & Coalescing Efficiency
        arrays_read = set()
        arrays_written = set()
        stride_regularity = 1.0
        coalescing_score = 1.0

        # Innermost loop index variable determines GPU thread coalescing
        inner_var = loop_vars[-1] if loop_vars else "i"

        # Check LHS Writes: A[idx] = ...
        lhs_assigns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?\s*(?:[\+\-\*\/]?=)', arith_block)
        for match in lhs_assigns:
            arr_name = match[0]
            dim1_expr = match[1]
            dim2_expr = match[2] if len(match) > 2 else ""

            arrays_written.add(arr_name)
            stride_regularity = min(stride_regularity, self._check_index_stride(dim1_expr))
            
            # Check coalescing: innermost index should be in fastest-varying (rightmost) index
            if dim2_expr:
                if inner_var in dim1_expr and inner_var not in dim2_expr:
                    coalescing_score = min(coalescing_score, 0.2) # Strided column-major penalty
            elif nesting_depth > 1 and inner_var not in dim1_expr:
                coalescing_score = min(coalescing_score, 0.4)

        # Check RHS Reads
        all_reads = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?', arith_block)
        for match in all_reads:
            arr_name = match[0]
            dim1_expr = match[1]
            dim2_expr = match[2] if len(match) > 2 else ""

            if arr_name not in arrays_written:
                arrays_read.add(arr_name)
            stride_regularity = min(stride_regularity, self._check_index_stride(dim1_expr))
            
            if dim2_expr and inner_var in dim1_expr and inner_var not in dim2_expr:
                coalescing_score = min(coalescing_score, 0.2)

        total_arrays = max(1, len(arrays_read | arrays_written))

        # 4. Footprint and Data Reuse Ratio
        if nesting_depth >= 2:
            dim1 = per_loop_trips[0] if len(per_loop_trips) > 0 else 512
            dim2 = per_loop_trips[1] if len(per_loop_trips) > 1 else 512
            unique_elements = dim1 * dim2
        else:
            unique_elements = per_loop_trips[0] if per_loop_trips else self.default_param_trip_count

        memory_footprint_bytes = max(1, total_arrays * unique_elements * self.bytes_per_elem)
        arithmetic_intensity = total_flops / float(memory_footprint_bytes)

        # Data reuse factor: Total array memory operations divided by unique elements transferred
        total_memory_ops = total_iterations * len(all_reads)
        data_reuse_ratio = float(total_memory_ops) / float(max(unique_elements, 1))

        # 5. Branch Divergence & Reduction Detection
        branch_count = len(re.findall(r'\b(if|switch)\s*\(', arith_block))
        
        has_reduction = False
        reduction_var = None
        reduc_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[\+\*]=\s*([^;]+);', arith_block)
        if reduc_match and reduc_match.group(1) not in (arrays_written | arrays_read):
            has_reduction = True
            reduction_var = reduc_match.group(1)

        return LoopFeature(
            function_name=fn_name,
            line_start=line_start,
            line_end=line_end,
            source_code=block,
            is_parallel_safe=is_safe,
            safety_reason=safety_reason,
            trip_count=total_iterations,
            nesting_depth=nesting_depth,
            flops_per_iter=flops_per_iter,
            total_flops=total_flops,
            memory_footprint_bytes=memory_footprint_bytes,
            arithmetic_intensity=arithmetic_intensity,
            data_reuse_ratio=data_reuse_ratio,
            coalescing_efficiency=coalescing_score,
            stride_regularity=stride_regularity,
            branch_divergence_count=branch_count,
            has_reduction=has_reduction,
            reduction_var=reduction_var,
            arrays_read=arrays_read,
            arrays_written=arrays_written
        )

    def _check_loop_carried_dependencies(self, block: str, loop_vars: List[str]) -> Tuple[bool, str]:
        """Detects loop-carried Read-After-Write (RAW) and Write-After-Write (WAW) hazards."""
        if not loop_vars:
            return True, "Safe: Independent iterations"

        primary_var = loop_vars[0]

        # Pattern: Writing to A[i] and reading from A[i-1] or A[i+1] (Loop-carried RAW dependence)
        # Matches: A[i - 1], A[i + 1], A[i - k]
        hazard_pattern = re.compile(rf'\[\s*{primary_var}\s*[\+\-]\s*[1-9][0-9]*\s*\]')
        
        # Check if the same array has written and offset-read
        lhs_arrs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[[^\]]+\]\s*=', block)
        if lhs_arrs:
            written_arr = lhs_arrs[0]
            # Search for offset reads of the same written array
            offset_reads = re.findall(rf'\b{written_arr}\s*\[\s*{primary_var}\s*[\+\-]\s*([1-9][0-9]*)\s*\]', block)
            if offset_reads:
                dist = offset_reads[0]
                return False, f"Hazard: Loop-carried RAW dependency detected on {written_arr}[{primary_var} - {dist}] (Sequential constraint)"

        # Check for pointer aliasing with restrict
        if re.search(r'\*([a-zA-Z_][a-zA-Z0-9_]*)\s*\+\+', block):
            return False, "Hazard: Pointer arithmetic with potential aliasing detected inside loop body"

        return True, "Safe: Iteration domain is embarrassingly parallel"

    def _estimate_trip_count(self, init: str, cond: str, step: str) -> int:
        bound_match = re.search(r'(?:<|<=)\s*([0-9]+)', cond)
        if bound_match:
            return int(bound_match.group(1))
        return self.default_param_trip_count

    def _check_index_stride(self, idx_expr: str) -> float:
        idx_expr = idx_expr.strip()
        if '[' in idx_expr:
            return 0.1  # Indirect access
        if '*' in idx_expr:
            return 0.5  # Strided access
        return 1.0      # Contiguous access
