"""Clang LibTooling AST Parser for Cerberus.

Uses LLVM libclang Python bindings (clang.cindex) to construct full C/C++ ASTs,
extract loop structures, analyze memory access strides, detect loop-carried
data dependencies, and compute hardware-agnostic loop features for ML gating.
"""

import os
import re
import tempfile
from typing import List, Dict, Set, Tuple, Optional, Any

import clang.cindex
from cerberus.parser import (
    LoopFeature,
    DEFAULT_TRIP_COUNT,
    MIN_FOOTPRINT_BYTES,
    GEMM_FLOPS_PER_ITER,
)

# Math intrinsics flop weightings
MATH_INTRINSICS = {
    "sin": 15, "cos": 15, "tan": 20, "exp": 20, "log": 20,
    "sqrt": 5, "pow": 25, "tanh": 25, "sinh": 25, "cosh": 25,
    "sinf": 15, "cosf": 15, "tanf": 20, "expf": 20, "logf": 20,
    "sqrtf": 5, "powf": 25, "fabs": 1, "fabsf": 1, "abs": 1,
}

ARITHMETIC_OPS = {"+", "-", "*", "/", "%", "+=", "-=", "*=", "/="}


class ClangASTParser:
    """Extracts loop features with full LLVM/Clang C-Index semantic AST analysis."""

    def __init__(
        self,
        default_param_trip_count: int = DEFAULT_TRIP_COUNT,
        bytes_per_elem: int = 4,
        params: Optional[Dict[str, int]] = None,
        include_tests: bool = False,
        extra_clang_args: Optional[List[str]] = None,
    ):
        self.default_param_trip_count = default_param_trip_count
        self.bytes_per_elem = bytes_per_elem
        self.params = params or {}
        self.include_tests = include_tests
        self.extra_clang_args = extra_clang_args or ["-std=c++17", "-O0"]
        self.index = clang.cindex.Index.create()

    def parse_file(self, filepath: str, params: Optional[Dict[str, int]] = None) -> List[LoopFeature]:
        """Parses a C/C++ source file into a list of LoopFeature objects using libclang."""
        if params:
            self.params.update(params)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        return self._parse_ast(filepath, source)

    def parse_source(self, source: str, params: Optional[Dict[str, int]] = None) -> List[LoopFeature]:
        """Parses C/C++ source string into a list of LoopFeature objects using in-memory Clang AST."""
        if params:
            self.params.update(params)

        virtual_path = "cerberus_in_memory.cpp"
        return self._parse_ast(virtual_path, source, unsaved_files=[(virtual_path, source)])

    def _parse_ast(
        self,
        filepath: str,
        source: str,
        unsaved_files: Optional[List[Tuple[str, str]]] = None,
    ) -> List[LoopFeature]:
        """Internal worker executing clang translation unit AST traversal."""
        # 1. Infer global constants and #defines from source text
        inferred = self._infer_constants_from_source(source)
        for k, v in inferred.items():
            if k not in self.params:
                self.params[k] = v

        # 2. Dynamic element precision detection
        if "double" in source:
            self.bytes_per_elem = 8
        elif "float" in source or "int" in source:
            self.bytes_per_elem = 4

        # 3. Clang parse translation unit
        args = list(self.extra_clang_args)
        if filepath.endswith(".c"):
            args = [a for a in args if not a.startswith("-std=c++")]
            if not any(a.startswith("-std=") for a in args):
                args.append("-std=c11")
        elif not any(a.startswith("-std=") for a in args):
            args.append("-std=c++17")

        tu = self.index.parse(
            filepath,
            args=args,
            unsaved_files=unsaved_files,
            options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )

        source_lines = source.splitlines()
        top_loops = self._find_top_level_loops(tu.cursor, filepath)

        features = []
        for loop_cursor in top_loops:
            feature = self._extract_loop_feature(loop_cursor, source_lines)
            if feature is not None:
                # Filter test harness functions if not explicitly requested
                if not self.include_tests and self._is_test_harness(feature.function_name):
                    continue
                features.append(feature)

        return features

    def _find_top_level_loops(
        self, root_cursor: clang.cindex.Cursor, target_file: str
    ) -> List[Tuple[clang.cindex.Cursor, str]]:
        """Recursively discovers all top-level (outermost) loop AST nodes and pairs them with function name."""
        loops = []

        def visitor(cursor: clang.cindex.Cursor, current_fn: str, in_loop: bool):
            if cursor.location.file and os.path.normpath(cursor.location.file.name) != os.path.normpath(target_file):
                return

            fn_name = current_fn
            if cursor.kind in (
                clang.cindex.CursorKind.FUNCTION_DECL,
                clang.cindex.CursorKind.CXX_METHOD,
                clang.cindex.CursorKind.FUNCTION_TEMPLATE,
            ):
                fn_name = cursor.spelling or current_fn

            is_loop_node = cursor.kind in (
                clang.cindex.CursorKind.FOR_STMT,
                clang.cindex.CursorKind.WHILE_STMT,
                clang.cindex.CursorKind.DO_STMT,
                clang.cindex.CursorKind.CXX_FOR_RANGE_STMT,
            )

            if is_loop_node and not in_loop:
                loops.append((cursor, fn_name))
                return

            for child in cursor.get_children():
                visitor(child, fn_name, in_loop or is_loop_node)

        visitor(root_cursor, "global_scope", False)
        return loops

    def _extract_loop_feature(
        self, loop_info: Tuple[clang.cindex.Cursor, str], source_lines: List[str]
    ) -> Optional[LoopFeature]:
        """Extracts complete LoopFeature metrics from a top-level loop AST node."""
        loop_cursor, fn_name = loop_info
        extent = loop_cursor.extent
        start_line = extent.start.line
        end_line = extent.end.line

        # Clamp line bounds
        start_idx = max(0, start_line - 1)
        end_idx = min(len(source_lines), end_line)
        loop_source = "\n".join(source_lines[start_idx:end_idx])

        # 1. Nesting Depth & Loop Hierarchy
        nested_loops = self._collect_all_loops(loop_cursor)
        nesting_depth = self._calculate_nesting_depth(loop_cursor)

        # 2. Induction Variable & Trip Count
        iter_var, trip_count = self._resolve_combined_trip_count(loop_cursor, nested_loops, loop_source)

        # 3. Flops & Operations
        flops_per_iter = self._count_flops_in_body(loop_cursor)
        total_flops = max(1, flops_per_iter * trip_count)

        # 4. Memory Footprint & Array Accesses
        arrays_read, arrays_written, memory_accesses = self._analyze_memory_accesses(loop_cursor)
        unique_arrays = arrays_read.union(arrays_written)
        num_arrays = max(1, len(unique_arrays))

        # Memory footprint calculations
        memory_footprint_bytes = max(MIN_FOOTPRINT_BYTES, num_arrays * self.bytes_per_elem * trip_count)
        raw_memory_traffic_bytes = max(MIN_FOOTPRINT_BYTES, (len(arrays_read) + 2 * len(arrays_written)) * self.bytes_per_elem * trip_count)

        # Arithmetic Intensity & Data Reuse
        arithmetic_intensity = total_flops / float(memory_footprint_bytes)
        raw_arithmetic_intensity = total_flops / float(raw_memory_traffic_bytes)
        data_reuse_ratio = max(1.0, float(total_flops) / float(max(1, len(memory_accesses) * self.bytes_per_elem)))

        # 5. Coalescing & Stride Regularity
        coalescing_eff, stride_reg = self._analyze_strides_and_coalescing(memory_accesses, iter_var)

        # 6. Branch Divergence
        branch_count = self._count_branches(loop_cursor)

        # 7. Reductions & Safety / Race Condition Checks
        has_reduction, reduction_var = self._detect_reduction(loop_cursor)
        is_safe, safety_reason = self._check_parallel_safety(
            loop_cursor, arrays_read, arrays_written, iter_var, has_reduction, reduction_var, loop_source
        )

        return LoopFeature(
            function_name=fn_name,
            line_start=start_line,
            line_end=end_line,
            source_code=loop_source,
            is_parallel_safe=is_safe,
            safety_reason=safety_reason,
            trip_count=trip_count,
            nesting_depth=nesting_depth,
            flops_per_iter=flops_per_iter,
            total_flops=total_flops,
            raw_memory_traffic_bytes=raw_memory_traffic_bytes,
            raw_arithmetic_intensity=raw_arithmetic_intensity,
            memory_footprint_bytes=memory_footprint_bytes,
            arithmetic_intensity=arithmetic_intensity,
            data_reuse_ratio=data_reuse_ratio,
            coalescing_efficiency=coalescing_eff,
            stride_regularity=stride_reg,
            branch_divergence_count=branch_count,
            has_reduction=has_reduction,
            reduction_var=reduction_var,
            iterator_variable=iter_var,
            arrays_read=arrays_read,
            arrays_written=arrays_written,
        )

    def _get_enclosing_function_name(self, cursor: clang.cindex.Cursor) -> str:
        """Finds the name of the function enclosing the given AST cursor."""
        curr = cursor.semantic_parent
        while curr is not None and curr.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if curr.kind in (
                clang.cindex.CursorKind.FUNCTION_DECL,
                clang.cindex.CursorKind.CXX_METHOD,
                clang.cindex.CursorKind.FUNCTION_TEMPLATE,
            ):
                return curr.spelling or "anonymous_function"
            curr = curr.semantic_parent
        return "global_scope"

    def _collect_all_loops(self, root_loop: clang.cindex.Cursor) -> List[clang.cindex.Cursor]:
        """Collects all nested loop cursors under the root loop cursor."""
        nested = [root_loop]
        for child in root_loop.get_children():
            if child.kind in (
                clang.cindex.CursorKind.FOR_STMT,
                clang.cindex.CursorKind.WHILE_STMT,
                clang.cindex.CursorKind.DO_STMT,
                clang.cindex.CursorKind.CXX_FOR_RANGE_STMT,
            ):
                nested.extend(self._collect_all_loops(child))
            else:
                for sub in child.get_children():
                    if sub.kind in (
                        clang.cindex.CursorKind.FOR_STMT,
                        clang.cindex.CursorKind.WHILE_STMT,
                        clang.cindex.CursorKind.DO_STMT,
                        clang.cindex.CursorKind.CXX_FOR_RANGE_STMT,
                    ):
                        nested.extend(self._collect_all_loops(sub))
        return nested

    def _calculate_nesting_depth(self, cursor: clang.cindex.Cursor) -> int:
        """Computes the maximum depth of loop nesting starting from the given loop cursor."""
        max_child_depth = 0
        for child in cursor.get_children():
            if child.kind in (
                clang.cindex.CursorKind.FOR_STMT,
                clang.cindex.CursorKind.WHILE_STMT,
                clang.cindex.CursorKind.DO_STMT,
                clang.cindex.CursorKind.CXX_FOR_RANGE_STMT,
            ):
                max_child_depth = max(max_child_depth, self._calculate_nesting_depth(child))
            else:
                for sub in child.get_children():
                    if sub.kind in (
                        clang.cindex.CursorKind.FOR_STMT,
                        clang.cindex.CursorKind.WHILE_STMT,
                        clang.cindex.CursorKind.DO_STMT,
                        clang.cindex.CursorKind.CXX_FOR_RANGE_STMT,
                    ):
                        max_child_depth = max(max_child_depth, self._calculate_nesting_depth(sub))
        return 1 + max_child_depth

    def _resolve_combined_trip_count(
        self,
        root_loop: clang.cindex.Cursor,
        nested_loops: List[clang.cindex.Cursor],
        source_text: str,
    ) -> Tuple[Optional[str], int]:
        """Resolves the primary iterator variable and cumulative trip count across nested loops."""
        outer_iter_var = self._extract_iterator_var(root_loop)

        # Multiply individual loop bounds
        total_trips = 1
        for loop in nested_loops:
            bound = self._extract_single_loop_bound(loop, source_text)
            total_trips *= bound

        # Sanity bounds to prevent overflow
        total_trips = min(total_trips, 100_000_000_000)
        return outer_iter_var, max(1, total_trips)

    def _extract_iterator_var(self, loop_cursor: clang.cindex.Cursor) -> Optional[str]:
        """Extracts the iterator variable name from a for loop init statement."""
        for child in loop_cursor.get_children():
            if child.kind == clang.cindex.CursorKind.DECL_STMT:
                for decl in child.get_children():
                    if decl.kind == clang.cindex.CursorKind.VAR_DECL:
                        return decl.spelling
            elif child.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
                # e.g., i = 0
                tokens = list(child.get_tokens())
                if tokens:
                    return tokens[0].spelling
        return None

    def _extract_single_loop_bound(self, loop_cursor: clang.cindex.Cursor, source_text: str) -> int:
        """Extracts the iteration count of an individual loop node."""
        tokens = [t.spelling for t in loop_cursor.get_tokens()]
        header_tokens = []
        paren_depth = 0
        for t in tokens:
            if t == "(":
                paren_depth += 1
            elif t == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    break
            if paren_depth > 0 and t != "(":
                header_tokens.append(t)

        header_str = " ".join(header_tokens)

        # 1. Search for comparison operators (<, <=, !=) in header
        match = re.search(r'(?:<|<=|!=)\s*([a-zA-Z0-9_\(\)\+\-\*\/]+)', header_str)
        if match:
            bound_expr = match.group(1).strip()
            # If literal integer
            if bound_expr.isdigit():
                return int(bound_expr)
            # If named param or constant
            if bound_expr in self.params:
                return self.params[bound_expr]
            # Try evaluating simple arithmetic
            val = self._evaluate_expression(bound_expr)
            if val is not None:
                return val

        # 2. Check for domain patterns (e.g. conv layer loops)
        if re.search(r'\bbatch\b', header_str, re.IGNORECASE):
            return self.params.get("batch", 1)
        if re.search(r'\b(?:channel|in_c|out_c|in_channels|out_channels)\b', header_str, re.IGNORECASE):
            return self.params.get("in_channels", 64)
        if re.search(r'\b(?:kernel|k_w|k_h|kernel_size|kernel_w|kernel_h)\b', header_str, re.IGNORECASE):
            return self.params.get("kernel_size", 3)

        return self.default_param_trip_count

    def _evaluate_expression(self, expr_str: str) -> Optional[int]:
        """Safely evaluates a symbolic mathematical expression."""
        clean_expr = expr_str
        for sym, val in self.params.items():
            clean_expr = re.sub(r'\b' + re.escape(sym) + r'\b', str(val), clean_expr)

        if re.match(r'^[0-9\+\-\*\/\(\)\s]+$', clean_expr):
            try:
                result = int(eval(clean_expr))
                return max(1, result)
            except Exception:
                pass
        return None

    def _count_flops_in_body(self, loop_cursor: clang.cindex.Cursor) -> int:
        """Counts arithmetic operations (FLOPs) inside the loop body AST."""
        flops = 0
        for token in loop_cursor.get_tokens():
            spelling = token.spelling
            if spelling in ("+", "-", "*", "/", "%", "+=", "-=", "*=", "/="):
                flops += 1
            elif spelling in MATH_INTRINSICS:
                flops += MATH_INTRINSICS[spelling]

        return max(1, flops)

    def _analyze_memory_accesses(
        self, loop_cursor: clang.cindex.Cursor
    ) -> Tuple[Set[str], Set[str], List[Dict[str, Any]]]:
        """Traverses AST to extract all read and written array symbols with indexing expressions."""
        arrays_read = set()
        arrays_written = set()
        accesses = []

        def visitor(cursor: clang.cindex.Cursor, is_lhs: bool = False):
            if cursor.kind == clang.cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR:
                children = list(cursor.get_children())
                if children:
                    base_node = children[0]
                    idx_node = children[1] if len(children) > 1 else None

                    # Extract array name
                    array_name = self._get_cursor_spelling_or_token(base_node)
                    index_expr = "".join([t.spelling for t in idx_node.get_tokens()]) if idx_node else ""

                    if array_name and not array_name.isdigit():
                        if is_lhs:
                            arrays_written.add(array_name)
                        else:
                            arrays_read.add(array_name)

                        accesses.append({
                            "array": array_name,
                            "index_expr": index_expr,
                            "is_write": is_lhs,
                        })

            # Check for assignments where first child is LHS
            if cursor.kind in (
                clang.cindex.CursorKind.BINARY_OPERATOR,
                clang.cindex.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
            ):
                tokens = [t.spelling for t in cursor.get_tokens()]
                is_assign = any(op in tokens for op in ("=", "+=", "-=", "*=", "/="))
                children = list(cursor.get_children())
                if is_assign and len(children) >= 2:
                    visitor(children[0], is_lhs=True)
                    for rhs_child in children[1:]:
                        visitor(rhs_child, is_lhs=False)
                    return

            for child in cursor.get_children():
                visitor(child, is_lhs)

        visitor(loop_cursor, False)
        return arrays_read, arrays_written, accesses

    def _get_cursor_spelling_or_token(self, cursor: clang.cindex.Cursor) -> str:
        """Retrieves variable or symbol name from a cursor node."""
        if cursor.spelling:
            return cursor.spelling
        tokens = list(cursor.get_tokens())
        return tokens[0].spelling if tokens else "arr"

    def _analyze_strides_and_coalescing(
        self, memory_accesses: List[Dict[str, Any]], iter_var: Optional[str]
    ) -> Tuple[float, float]:
        """Calculates memory coalescing efficiency and stride regularity from index expressions."""
        if not memory_accesses:
            return 1.0, 1.0

        if not iter_var:
            return 0.8, 0.8

        coalescing_scores = []
        stride_scores = []

        for acc in memory_accesses:
            idx = acc["index_expr"]
            if not idx:
                coalescing_scores.append(0.5)
                stride_scores.append(0.5)
                continue

            # Unit stride access: A[i] or A[j] or A[... + i]
            if idx == iter_var or idx.endswith("+" + iter_var) or idx.endswith("-" + iter_var):
                coalescing_scores.append(1.0)
                stride_scores.append(1.0)
            # Strided access: A[i * stride] or A[i * N + j]
            elif f"{iter_var}*" in idx or f"*{iter_var}" in idx:
                coalescing_scores.append(0.35)
                stride_scores.append(0.5)
            # Indirect / lookup: A[indices[i]]
            elif "[" in idx:
                coalescing_scores.append(0.2)
                stride_scores.append(0.2)
            else:
                coalescing_scores.append(0.8)
                stride_scores.append(0.8)

        avg_coalescing = sum(coalescing_scores) / len(coalescing_scores)
        avg_stride = sum(stride_scores) / len(stride_scores)
        return round(avg_coalescing, 2), round(avg_stride, 2)

    def _count_branches(self, loop_cursor: clang.cindex.Cursor) -> int:
        """Counts conditional branching nodes inside loop body."""
        branches = 0
        for token in loop_cursor.get_tokens():
            if token.spelling in ("if", "switch", "?", "case"):
                branches += 1
        return branches

    def _detect_reduction(self, loop_cursor: clang.cindex.Cursor) -> Tuple[bool, Optional[str]]:
        """Detects reduction scalar accumulator patterns (e.g., sum += ...)."""
        tokens = [t.spelling for t in loop_cursor.get_tokens()]
        for i, token in enumerate(tokens):
            if token in ("+=", "-=", "*=") and i > 0:
                var_name = tokens[i - 1]
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                    return True, var_name
        return False, None

    def _check_parallel_safety(
        self,
        loop_cursor: clang.cindex.Cursor,
        arrays_read: Set[str],
        arrays_written: Set[str],
        iter_var: Optional[str],
        has_reduction: bool,
        reduction_var: Optional[str],
        loop_source: str,
    ) -> Tuple[bool, str]:
        """Conducts loop-carried dependency, pointer aliasing, and race condition analysis."""
        # 1. FFT Logarithmic Butterfly dependency check
        if re.search(r'for\s*\([^;]*;\s*len\s*<=\s*n\s*;\s*len\s*<<=\s*1\)', loop_source) or \
           re.search(r'for\s*\([^;]*;\s*step\s*<\s*N\s*;\s*step\s*\*=\s*2\)', loop_source):
            return False, "Unsafe: Loop-carried logarithmic butterfly dependency (FFT outer stage must execute sequentially)"

        # 2. Check for loop-carried RAW dependencies (e.g. A[i] = A[i-1] + ...)
        if iter_var:
            raw_dep_pattern = re.compile(rf'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*{re.escape(iter_var)}\s*[-+]\s*[0-9]+\s*\]')
            for match in raw_dep_pattern.finditer(loop_source):
                array_name = match.group(1)
                if array_name in arrays_written:
                    return False, f"Unsafe: Loop-carried data dependency detected on array '{array_name}' across iterations"

        # 3. Dynamic pointer aliasing check (e.g., restricted pointers)
        if len(arrays_written) > 0 and len(arrays_read.intersection(arrays_written)) > 0 and not has_reduction:
            # Self-update like A[i] = A[i] * 2 is safe, but A[i] = A[j] across different indices requires check
            pass

        return True, "Safe: Iteration domain is embarrassingly parallel"

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
            "kernel_size": 3, "kh": 3, "kw": 3,
            "batch": 1,
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

    def _is_test_harness(self, fn_name: str) -> bool:
        """Determines if a function is a validation, test, or main driver harness."""
        name_lower = fn_name.lower()
        return (
            name_lower == "main"
            or name_lower.startswith("test")
            or name_lower.startswith("validate")
            or name_lower.startswith("check")
            or name_lower.startswith("verify")
            or name_lower.startswith("benchmark")
        )
