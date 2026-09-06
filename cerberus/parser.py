"""AST Loop Feature Extractor for C and C++ Source Code."""

import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

# Named constants replacing previously hardcoded values
MIN_FOOTPRINT_BYTES = 1024       # Floor to prevent division-by-zero in AI calculation
DEFAULT_TRIP_COUNT = 10000       # Realistic HPC workload default for unresolved variables
GEMM_FLOPS_PER_ITER = 2          # Canonical GEMM inner loop: 1 FMA = 2 FLOPs (mul + add)

@dataclass
class LoopFeature:
    function_name: str
    line_start: int
    line_end: int
    source_code: str
    is_parallel_safe: bool
    safety_reason: str
    trip_count: int
    nesting_depth: int
    flops_per_iter: int
    total_flops: int
    raw_memory_traffic_bytes: int
    raw_arithmetic_intensity: float
    memory_footprint_bytes: int
    arithmetic_intensity: float
    data_reuse_ratio: float
    coalescing_efficiency: float
    stride_regularity: float
    branch_divergence_count: int
    has_reduction: bool
    reduction_var: Optional[str]
    iterator_variable: Optional[str]
    arrays_read: Set[str]
    arrays_written: Set[str]

    @property
    def raw_source(self) -> str:
        return self.source_code

    def to_feature_vector(self) -> List[float]:
        """Convert loop metrics to normalized ML input feature vector."""
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
            1.0 if self.has_reduction else 0.0,
        ]


class CLoopParser:
    """Extracts loop features from C and C++ source code using hierarchical block analysis."""

    def __init__(self, default_param_trip_count: int = DEFAULT_TRIP_COUNT, bytes_per_elem: int = 4, params: Optional[Dict[str, int]] = None, include_tests: bool = False):
        self.default_param_trip_count = default_param_trip_count
        self.bytes_per_elem = bytes_per_elem
        self.params = params or {}
        self.include_tests = include_tests

    def parse_file(self, filepath: str, params: Optional[Dict[str, int]] = None) -> List[LoopFeature]:
        if params:
            self.params.update(params)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return self.parse_source(content, params=params)

    def parse_source(self, source: str, params: Optional[Dict[str, int]] = None) -> List[LoopFeature]:
        """Finds all top-level loop regions in the source code and extracts features with call-site inference."""
        if params:
            self.params.update(params)
        lines = source.splitlines()
        loop_features = []
        
        # 1. Interprocedural Constant & Size Inference (extract #define, main(), and global declarations)
        inferred_params = self._infer_constants_from_source(source)
        for k, v in inferred_params.items():
            if k not in self.params:
                self.params[k] = v

        # Dynamic array element precision
        self.bytes_per_elem = self._detect_bytes_per_elem(source)

        # Track active function name across multiline declarations
        fn_matches = []
        fn_decl_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;{]*\)\s*(?:const)?\s*\{', re.MULTILINE)
        for m in fn_decl_pattern.finditer(source):
            fn_name = m.group(1)
            if fn_name not in ("for", "if", "while", "switch", "catch"):
                line_no = source[:m.start()].count('\n')
                fn_matches.append((line_no, fn_name))

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Update current enclosing function
            current_fn = "unknown"
            for fn_line, fn_name in fn_matches:
                if fn_line <= i:
                    current_fn = fn_name

            # Filter out validation/test harness functions and main() unless explicitly enabled
            if not self.include_tests and (current_fn.startswith(("validate_", "check_", "assert_", "benchmark_test_")) or current_fn == "main"):
                i += 1
                continue

            # Look for top-level for loops
            if re.search(r'\bfor\s*\(', line):
                loop_block, end_line = self._extract_block(lines, i)
                feature = self._analyze_loop_region(loop_block, i + 1, end_line + 1, current_fn)
                if feature:
                    loop_features.append(feature)
                i = end_line
            # Look for while loops (excluding do-while tails)
            elif re.search(r'\bwhile\s*\(', line) and not line.endswith(';'):
                loop_block, end_line = self._extract_block(lines, i)
                feature = self._analyze_while_loop_region(loop_block, i + 1, end_line + 1, current_fn)
                if feature:
                    loop_features.append(feature)
                i = end_line
            i += 1

        return loop_features

    def _detect_bytes_per_elem(self, text: str) -> int:
        """Determines tensor element precision (8B for double/int64, 4B for float/int32, 1B for char)."""
        if re.search(r'\b(double|int64_t|long\s+long|uint64_t)\b', text):
            return 8
        elif re.search(r'\b(char|uint8_t|int8_t|bool)\b', text):
            return 1
        elif re.search(r'\b(short|int16_t|uint16_t|half)\b', text):
            return 2
        return 4

    def _infer_constants_from_source(self, source: str) -> Dict[str, int]:
        """Scans caller scopes, #define macros, and declarations to automatically resolve unbound dynamic variables."""
        inferred = {}

        # Strip validation / unit-test toy scopes (e.g. void validate_*() { const int N = 8; ... })
        clean_source = re.sub(r'void\s+(?:validate|test|check)_[a-zA-Z0-9_]*\s*\([^)]*\)\s*\{[^}]*\}', '', source)
        
        # Pattern 0: #define VAR_NAME VALUE / EXPR
        define_matches = re.findall(r'#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([^\r\n]+)', clean_source)
        for var_name, val_expr in define_matches:
            val_expr = val_expr.split('//')[0].strip()
            clean_def = re.sub(r'[^0-9\+\-\*\/\(\)\s]', '', val_expr).strip()
            if clean_def:
                try:
                    val = int(eval(clean_def, {"__builtins__": None}, {}))
                    if val >= 1:
                        inferred[var_name] = val
                except Exception:
                    pass

        # Pattern 1: constexpr / const / int VAR = VALUE; (outside validation tests)
        const_matches = re.findall(r'(?:constexpr|const|static\s+const|int|size_t|auto)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]+);', clean_source)
        for var_name, val_str in const_matches:
            val_str = val_str.split('//')[0].strip()
            clean_val = re.sub(r'[^0-9\+\-\*\/\(\)\s]', '', val_str).strip()
            if clean_val:
                try:
                    val = int(eval(clean_val, {"__builtins__": None}, {}))
                    if val >= 32 or var_name not in inferred:
                        inferred[var_name] = val
                except Exception:
                    pass

        # Pattern 2: std::vector<float> data(SIZE, ...) or std::vector<int> data(SIZE)
        vec_matches = re.findall(r'std::vector<[^>]+>\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*([a-zA-Z0-9_\* \+\-]+?)\s*(?:,|\))', clean_source)
        for expr in vec_matches:
            expr_clean = expr.strip()
            if expr_clean.isdigit():
                inferred["size"] = int(expr_clean)
                inferred["N"] = int(expr_clean)

        # Pattern 3: Common dimensional aliases and domain defaults
        if "MATRIX_N" in inferred:
            inferred["N"] = inferred["MATRIX_N"]
            inferred["dim"] = inferred["MATRIX_N"]
        if "WIDTH" in inferred:
            inferred["width"] = inferred["WIDTH"]
        if "HEIGHT" in inferred:
            inferred["height"] = inferred["HEIGHT"]

        # Domain-aware dimension defaults for unresolved tensor/convolution/image variables
        domain_defaults = {
            "kernel_size": 3, "k": 3, "kh": 3, "kw": 3,
            "batch": 1, "b": 1,
            "in_channels": 64, "ic": 64,
            "out_channels": 64, "oc": 64,
            "height": 224, "width": 224, "h": 224, "w": 224,
            "out_h": 222, "out_w": 222,
            "rows": 1024, "cols": 1024,
            "num_particles": 10000, "dims": 3,
            "intensity": 100,
        }
        for k, v in domain_defaults.items():
            if k not in inferred:
                inferred[k] = v

        # Default minimum production scale for unspecified N
        if "N" not in inferred or inferred["N"] < 32:
            inferred["N"] = self.default_param_trip_count

        return inferred

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

    def _analyze_loop_region(self, block: str, line_start: int, line_end: int, fn_name: str) -> Optional[LoopFeature]:
        """Hierarchically analyzes a loop nest without incorrectly multiplying sibling loops."""
        
        # 1. Parse Loop Nest Hierarchy
        loop_nest_info = self._extract_hierarchical_loops(block)
        if not loop_nest_info:
            return None

        nesting_depth = loop_nest_info["max_depth"]
        total_iterations = loop_nest_info["total_iterations"]
        loop_vars = loop_nest_info["loop_vars"]
        outer_trip = loop_nest_info["outer_trip"]
        inner_trip = loop_nest_info["inner_trip"]

        # 2. Extract Arithmetic & Memory Operations in Loop Body
        arith_block = re.sub(r'for\s*\([^)]*\)', '', block)

        # Safety & Dependency Check
        is_safe, safety_reason = self._check_loop_carried_dependencies(arith_block, loop_vars)

        # FLOP Operational Count per iteration
        add_subs = len(re.findall(r'(?<!\+)\+(?!\+) | (?<!\-)\-(?!\-)', arith_block, re.VERBOSE))
        muls = len(re.findall(r'\*', arith_block))
        divs = len(re.findall(r'\/', arith_block))
        math_calls = len(re.findall(r'\b(?:std::)?(exp|log|sin|cos|sqrt|pow|fma|tanh|fabs|abs)f?\b', arith_block))
        
        # In nested GEMM/Matmul (depth 3), check if inner body matches FMA pattern: sum += A[...] * B[...]
        if nesting_depth >= 3:
            fma_pattern = re.search(r'\+\=.*\*', arith_block)
            if fma_pattern:
                flops_per_iter = GEMM_FLOPS_PER_ITER  # Canonical GEMM: 1 mul + 1 add = 2 FLOPs
            else:
                flops_per_iter = max(1, add_subs + muls + (divs * 4) + (math_calls * 10))
        else:
            flops_per_iter = max(1, add_subs + muls + (divs * 4) + (math_calls * 10))
            
        total_flops = total_iterations * flops_per_iter

        # 3. Array Access & Memory Pattern Analysis
        arrays_read = set()
        arrays_written = set()
        stride_regularity = 1.0
        coalescing_score = 1.0

        inner_var = loop_vars[-1] if loop_vars else "i"
        outer_var = loop_vars[0] if loop_vars else "i"

        # Check Array Writes (LHS): A[idx] = ...
        lhs_assigns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?\s*([\+\-\*\/]?=)', arith_block)
        for match in lhs_assigns:
            arr_name = match[0]
            dim1_expr = match[1]
            dim2_expr = match[2] if len(match) > 2 else ""
            op = match[3] if len(match) > 3 else "="

            arrays_written.add(arr_name)
            if op in ("+=", "-=", "*=", "/="):
                arrays_read.add(arr_name) # Compound assignment reads prior value

            stride_regularity = min(stride_regularity, self._check_index_stride(dim1_expr, inner_var))
            coalescing_score = min(coalescing_score, self._check_coalescing(dim1_expr, dim2_expr, inner_var, nesting_depth))

        # Check Array Reads (RHS)
        all_reads = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?', arith_block)
        for match in all_reads:
            arr_name = match[0]
            dim1_expr = match[1]
            dim2_expr = match[2] if len(match) > 2 else ""

            arrays_read.add(arr_name)
            stride_regularity = min(stride_regularity, self._check_index_stride(dim1_expr, inner_var))
            coalescing_score = min(coalescing_score, self._check_coalescing(dim1_expr, dim2_expr, inner_var, nesting_depth))


        total_arrays = max(1, len(arrays_read | arrays_written))
        raw_accesses_per_iter = max(1, len(lhs_assigns) + len(all_reads))
        raw_memory_traffic_bytes = total_iterations * raw_accesses_per_iter * self.bytes_per_elem
        raw_arithmetic_intensity = total_flops / float(max(1, raw_memory_traffic_bytes))

        # 4. Mathematically Rigorous Memory Footprint & Data Reuse
        if nesting_depth >= 3:
            # 3D Matrix Multiplication: 3 distinct N x N matrices (A, B, C)
            matrix_dim = outer_trip
            num_matrices = max(3, len(arrays_read | arrays_written))
            unique_elements = num_matrices * (matrix_dim * matrix_dim)
            memory_footprint_bytes = max(1024, unique_elements * self.bytes_per_elem)
            # Effective FLOP/Byte = (2 * N^3) / (3 * N^2 * 4) = N / 6 FLOP/Byte
            arithmetic_intensity = total_flops / float(memory_footprint_bytes)
            data_reuse_ratio = float(matrix_dim)
        elif nesting_depth == 2:
            # 2D Stencil / Grid: Footprint = total_arrays * dim1 * dim2 * 4 bytes
            unique_elements = total_arrays * (outer_trip * inner_trip)
            memory_footprint_bytes = max(MIN_FOOTPRINT_BYTES, unique_elements * self.bytes_per_elem)
            arithmetic_intensity = total_flops / float(memory_footprint_bytes)
            data_reuse_ratio = float(len(all_reads))
        else:
            # 1D Vector: Footprint = total_arrays * outer_trip * 4 bytes
            unique_elements = total_arrays * outer_trip
            memory_footprint_bytes = max(MIN_FOOTPRINT_BYTES, unique_elements * self.bytes_per_elem)
            arithmetic_intensity = total_flops / float(memory_footprint_bytes)
            data_reuse_ratio = float(len(all_reads))

        # 5. Branch Divergence & True Parallel Reduction Detection
        branch_count = len(re.findall(r'\b(if|switch)\s*\(', arith_block))
        
        has_reduction = False
        reduction_var = None
        reduc_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[\+\*]=\s*([^;]+);', arith_block)
        if reduc_match:
            cand_var = reduc_match.group(1)
            # Check if this variable is reset/declared inside the loop body (making it thread-private)
            is_reset_inside = bool(re.search(rf'(?:float|int|double|auto)\s+{cand_var}\s*=|{cand_var}\s*=\s*0', arith_block))
            if not is_reset_inside and cand_var not in (arrays_written | arrays_read):
                has_reduction = True
                reduction_var = cand_var

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
            raw_memory_traffic_bytes=raw_memory_traffic_bytes,
            raw_arithmetic_intensity=raw_arithmetic_intensity,
            memory_footprint_bytes=memory_footprint_bytes,
            arithmetic_intensity=arithmetic_intensity,
            data_reuse_ratio=data_reuse_ratio,
            coalescing_efficiency=coalescing_score,
            stride_regularity=stride_regularity,
            branch_divergence_count=branch_count,
            has_reduction=has_reduction,
            reduction_var=reduction_var,
            iterator_variable=outer_var,
            arrays_read=arrays_read,
            arrays_written=arrays_written
        )

    def _analyze_while_loop_region(self, block: str, line_start: int, line_end: int, fn_name: str) -> Optional[LoopFeature]:
        """Analyzes a while loop construct, estimating trip counts and dependencies."""
        cond_match = re.search(r'while\s*\(([^)]+)\)', block)
        if not cond_match:
            return None

        cond_expr = cond_match.group(1).strip()
        trip_count = self._estimate_trip_count("", cond_expr, "")

        arith_block = re.sub(r'while\s*\([^)]*\)', '', block)

        # Loop variables heuristic from condition or increment in body
        iter_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\+\+|\-\-|\+=|-=|<|>|<=|>=)', block)
        iter_var = iter_vars[0] if iter_vars else "i"

        is_safe, safety_reason = self._check_loop_carried_dependencies(arith_block, [iter_var])

        add_subs = len(re.findall(r'(?<!\+)\+(?!\+) | (?<!\-)\-(?!\-)', arith_block, re.VERBOSE))
        muls = len(re.findall(r'\*', arith_block))
        divs = len(re.findall(r'\/', arith_block))
        math_calls = len(re.findall(r'\b(?:std::)?(exp|log|sin|cos|sqrt|pow|fma|tanh|fabs|abs)f?\b', arith_block))
        flops_per_iter = max(1, add_subs + muls + (divs * 4) + (math_calls * 10))
        total_flops = trip_count * flops_per_iter

        arrays_read = set()
        arrays_written = set()
        lhs_assigns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?\s*([\+\-\*\/]?=)', arith_block)
        for m in lhs_assigns:
            arrays_written.add(m[0])
            op = m[3] if len(m) > 3 else "="
            if op in ("+=", "-=", "*=", "/="):
                arrays_read.add(m[0])

        all_reads = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[([^\]]+)\](?:\s*\[([^\]]+)\])?', arith_block)
        for m in all_reads:
            arrays_read.add(m[0])

        # Analyze memory access patterns properly (same as for-loops)
        stride_regularity = 1.0
        coalescing_score = 1.0
        for m in lhs_assigns:
            arr_name = m[0]
            dim1_expr = m[1]
            dim2_expr = m[2] if len(m) > 2 else ""
            stride_regularity = min(stride_regularity, self._check_index_stride(dim1_expr, iter_var))
            coalescing_score = min(coalescing_score, self._check_coalescing(dim1_expr, dim2_expr, iter_var, 1))
        for m in all_reads:
            dim1_expr = m[1]
            dim2_expr = m[2] if len(m) > 2 else ""
            stride_regularity = min(stride_regularity, self._check_index_stride(dim1_expr, iter_var))
            coalescing_score = min(coalescing_score, self._check_coalescing(dim1_expr, dim2_expr, iter_var, 1))

        total_arrays = max(1, len(arrays_read | arrays_written))
        unique_elements = total_arrays * trip_count
        memory_footprint_bytes = max(MIN_FOOTPRINT_BYTES, unique_elements * self.bytes_per_elem)
        arithmetic_intensity = total_flops / float(memory_footprint_bytes)

        branch_count = len(re.findall(r'\b(if|switch)\s*\(', arith_block))

        return LoopFeature(
            function_name=fn_name,
            line_start=line_start,
            line_end=line_end,
            source_code=block,
            is_parallel_safe=is_safe,
            safety_reason=safety_reason,
            trip_count=trip_count,
            nesting_depth=1,
            flops_per_iter=flops_per_iter,
            total_flops=total_flops,
            raw_memory_traffic_bytes=trip_count * max(1, len(lhs_assigns) + len(all_reads)) * self.bytes_per_elem,
            raw_arithmetic_intensity=arithmetic_intensity,
            memory_footprint_bytes=memory_footprint_bytes,
            arithmetic_intensity=arithmetic_intensity,
            data_reuse_ratio=float(max(1, len(all_reads))),
            coalescing_efficiency=coalescing_score,
            stride_regularity=stride_regularity,
            branch_divergence_count=branch_count,
            has_reduction=False,
            reduction_var=None,
            iterator_variable=iter_var,
            arrays_read=arrays_read,
            arrays_written=arrays_written
        )

    def _extract_hierarchical_loops(self, block: str) -> Optional[Dict]:
        """Extracts true nesting depth and trips without multiplying sequential sibling loops."""
        for_pattern = re.compile(r'for\s*\(([^;:]*);([^;:]*);([^)]*)\)|for\s*\(([^:]+):([^)]+)\)')
        matches = list(for_pattern.finditer(block))
        if not matches:
            return None

        # Build scope depth for each for loop based on brace depth in block
        loop_vars = []
        trips = []

        for m in matches:
            init_part = m.group(1) or ""
            cond_part = m.group(2) or ""
            step_part = m.group(3) or ""
            range_decl = m.group(4) or ""

            if range_decl:
                var_m = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*$', range_decl.strip().rstrip('&').strip())
                loop_vars.append(var_m.group(1) if var_m else "elem")
                trips.append(self.default_param_trip_count)
            else:
                var_m = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', init_part)
                loop_vars.append(var_m.group(1) if var_m else "i")
                trips.append(self._estimate_trip_count(init_part, cond_part, step_part))

        # Check true nesting vs sibling loops
        # Count consecutive opening braces '{' between for loops
        is_strictly_nested = True
        for i in range(len(matches) - 1):
            sub_text = block[matches[i].end():matches[i+1].start()]
            # If a closing brace '}' occurs before the next loop, it's a sibling loop, not purely nested
            if '}' in sub_text:
                is_strictly_nested = False
                break

        if is_strictly_nested:
            max_depth = len(matches)
            total_iterations = 1
            for t in trips:
                total_iterations *= t
        else:
            # Outer loop contains sibling inner loops: iterations = outer_trip * sum(inner_trips)
            max_depth = 2
            outer_trip = trips[0]
            inner_sum = sum(trips[1:]) if len(trips) > 1 else 1
            total_iterations = outer_trip * inner_sum

        return {
            "max_depth": max_depth,
            "total_iterations": total_iterations,
            "loop_vars": loop_vars,
            "outer_trip": trips[0] if trips else 1000,
            "inner_trip": trips[1] if len(trips) > 1 else (trips[0] if trips else 1000)
        }

    def _check_coalescing(self, dim1_expr: str, dim2_expr: str, inner_var: str, depth: int) -> float:
        """Determines if GPU warp threads access contiguous memory addresses."""
        dim1 = dim1_expr.strip()
        dim2 = dim2_expr.strip()

        # 1. Indirect array gather: A[indices[i]] or A[expr % N]
        if '[' in dim1 or '%' in dim1 or 'static_cast' in dim1 or '*' in dim1 and inner_var not in dim1:
            return 0.15

        # 2. 2D Array: A[i][j] where inner_var is in rightmost dimension (dim2)
        if dim2:
            if inner_var in dim2:
                return 1.0  # Optimal row-major contiguous access
            elif inner_var in dim1:
                return 0.25 # Column-major strided access
            else:
                return 0.5

        # 3. 1D Array: A[i] where i is inner_var
        if dim1 == inner_var or dim1.endswith(f"+ {inner_var}") or dim1.endswith(f"- {inner_var}"):
            return 1.0 # 100% Contiguous Coalescing

        # 4. 1D Stencil: A[i + 1] or A[i - 1]
        if inner_var in dim1 and ('+' in dim1 or '-' in dim1) and '*' not in dim1:
            return 1.0 # Contiguous Stencil

        # 5. Strided 1D: A[i * 2] or A[i * N + j]
        if f"{inner_var} *" in dim1 or f"* {inner_var}" in dim1:
            return 0.35

        return 0.8

    def _check_index_stride(self, idx_expr: str, inner_var: str) -> float:
        """Calculates stride regularity score (1.0 = contiguous, 0.5 = strided, 0.1 = irregular gather)."""
        idx = idx_expr.strip()
        if '[' in idx or '%' in idx or 'static_cast' in idx:
            return 0.1  # Irregular Gather / Scatter
        if '*' in idx and inner_var in idx:
            return 0.5  # Strided Access
        return 1.0      # Contiguous Stride-1 Access

    def _check_loop_carried_dependencies(self, block: str, loop_vars: List[str]) -> Tuple[bool, str]:
        """Detects loop-carried Read-After-Write (RAW) hazards."""
        if not loop_vars:
            return True, "Safe: Independent iterations"

        primary_var = loop_vars[0]

        # 1. Logarithmic/butterfly stage loop detection (e.g., len <<= 1, len *= 2 in FFT / bitonic sort)
        # In multi-stage butterfly algorithms, stage N reads results written by stage N-1 in-place
        header_text = block[:block.find('{')] if '{' in block else block[:80]
        if re.search(r'\b[a-zA-Z_]\w*\s*(?:<<=|>>=|\*=|/=|<<|>>)', header_text):
            # Check if body modifies arrays in place that are re-read
            if re.search(r'\[[^\]]+\]\s*[\+\-\*\/]?=', block):
                return False, "Hazard: Logarithmic butterfly stage-carried dependency across sequential passes (Outer loop is serial)"

        # 2. Pattern: Writing to A[i] and reading from A[i-1] (Loop-carried RAW dependence)
        lhs_arrs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[[^\]]+\]\s*=', block)
        if lhs_arrs:
            written_arr = lhs_arrs[0]
            offset_reads = re.findall(rf'\b{written_arr}\s*\[\s*{primary_var}\s*[\+\-]\s*([1-9][0-9]*)\s*\]', block)
            if offset_reads:
                dist = offset_reads[0]
                return False, f"Hazard: Loop-carried RAW dependency detected on {written_arr}[{primary_var} - {dist}] (Sequential constraint)"

        # 3. Check for pointer aliasing with restrict
        if re.search(r'\*([a-zA-Z_][a-zA-Z0-9_]*)\s*\+\+', block):
            return False, "Hazard: Pointer arithmetic with potential aliasing detected inside loop body"

        return True, "Safe: Iteration domain is embarrassingly parallel"

    def _estimate_trip_count(self, init: str, cond: str, step: str) -> int:
        """Computes trip count = max(1, (stop - start) / step) for arbitrary loop variables."""
        start_val = 0
        start_match = re.search(r'=\s*([0-9]+)', init)
        if start_match:
            start_val = int(start_match.group(1))

        step_val = 1
        step_match = re.search(r'\+=\s*([0-9]+)|=\s*\w+\s*\+\s*([0-9]+)', step)
        if step_match:
            step_val = int(step_match.group(1) or step_match.group(2))

        # 3. Stop / bound value (e.g. x < 1000, k <= 512, i < N, i < N - 1, i < MATRIX_N)
        bound_match = re.search(r'(<|<=)\s*([^;]+)', cond)
        if bound_match:
            op = bound_match.group(1)
            raw_expr = bound_match.group(2).strip()

            # Substitute known params into expr
            eval_expr = raw_expr
            for p_name, p_val in self.params.items():
                eval_expr = re.sub(rf'\b{p_name}\b', str(p_val), eval_expr)

            # Replace dynamic size() with default N
            eval_expr = re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*\.size\(\)', str(self.params.get('size', self.params.get('N', self.default_param_trip_count))), eval_expr)
            
            # Strip C/C++ float literals and subscripts
            eval_expr = re.sub(r'\[[^\]]*\]', '', eval_expr)
            eval_expr = re.sub(r'(?<=[0-9])f\b', '', eval_expr)
            eval_expr = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', str(self.default_param_trip_count), eval_expr)

            # Keep only safe arithmetic characters (+ - * / ( ) digits whitespace)
            clean_math = re.sub(r'[^0-9\+\-\*\/\(\)\s]', '', eval_expr).strip()

            try:
                if clean_math:
                    stop_val = int(eval(clean_math, {"__builtins__": None}, {}))
                else:
                    stop_val = self.default_param_trip_count
            except Exception:
                stop_val = self.default_param_trip_count

            if op == '<=':
                stop_val += 1
            return max(1, (stop_val - start_val + (step_val - 1)) // step_val)

        return self.default_param_trip_count
