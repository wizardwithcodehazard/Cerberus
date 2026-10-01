# Security Policy

## Reporting Security Issues

If you discover a potential security vulnerability in Cerberus, please do not file a public issue. Instead, report it privately to the maintainers at `teamseeplusplus@gmail.com` or via GitHub Private Vulnerability Reporting.

---

## Static Code Analysis & Expression Evaluation

### Use of `eval()` in Macro Resolution
Cerberus performs static source-to-source analysis on C/C++ loop constructs. To compute symbolic trip counts from `#define` macros (e.g., `#define BLOCK_SIZE (64 * 4)`), Cerberus evaluates constant arithmetic expressions.

- **Sanitization:** Input expressions are strictly validated against a character whitelist (`^[0-9\+\-\*\/\(\)\s]+$`) before evaluation. Any expressions containing identifiers, function calls, or non-arithmetic characters are rejected without executing.
- **Scope:** Parsing is restricted to static analysis of local source files and does not execute compiled host or device binaries.
- **Guidance:** As with any compiler or static analysis tool, do not run Cerberus on untrusted or malicious source code in high-privilege execution environments.
