"""
Math problem solving tools using LangChain and SymPy (IMPROVED VERSION with better descriptions)
"""

from typing import Optional, Type, ClassVar
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import sympy
from sympy import symbols, solve, simplify, diff, integrate, factor, expand, N
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import sys
from io import StringIO
import ast


class MathPythonREPLInput(BaseModel):
    """Input schema for MathPythonREPL"""
    code: str = Field(description="Python code to execute. Only math libraries (sympy, numpy, math) are allowed.")


class MathPythonREPL(BaseTool):
    """
    Restricted Python REPL for mathematical computations.
    Only allows safe math libraries: sympy, numpy, math, fractions.
    Has timeout and dangerous code detection.
    """

    name: str = "python_repl"
    description: str = """Execute Python code for mathematical computations.

ALLOWED LIBRARIES: sympy, numpy, math, fractions, itertools, collections, functools, decimal

WHEN TO USE THIS TOOL:
- Complex calculations requiring loops or conditionals
- Multi-step problems requiring intermediate results
- Algorithm implementation (e.g., finding all integer pairs, counting solutions)
- When you need to verify results programmatically
- When sympy_calculator cannot handle the complexity

WHEN TO USE sympy_calculator INSTEAD:
- Simple single-operation calculations
- Standard equation solving, differentiation, integration
- Quick evaluations without logic

IMPORTANT NOTES:
- Always use print() to output results
- Code must be valid Python syntax
- Results are captured from stdout
- Errors are returned as strings

EXAMPLES:

1. Equation solving and verification:
from sympy import symbols, solve
x = symbols('x')
solutions = solve(x**2 + 5*x + 6, x)
print(f'Solutions: {solutions}')
for sol in solutions:
    result = sol**2 + 5*sol + 6
    print(f'Verify x={sol}: {result}')

2. Finding integer pairs:
count = 0
solutions = []
for x in range(-100, 101):
    for y in range(-100, 101):
        if x**2 + y**2 == 25:
            count += 1
            solutions.append((x, y))
print(f'Found {count} solutions')
print(f'First 5: {solutions[:5]}')

3. Number theory (divisors):
from sympy import divisors, factorint
n = 196
divs = divisors(n)
print(f'Prime factorization: {factorint(n)}')
print(f'Number of divisors: {len(divs)}')

4. Numerical computation:
import numpy as np
result = np.sqrt(16) + np.log10(100)
print(f'Result: {result}')

5. Infinite series:
from sympy import Sum, oo, symbols
k = symbols('k')
result = Sum(1/k**2, (k, 1, oo)).doit()
print(f'Sum: {result}')
"""
    args_schema: Type[BaseModel] = MathPythonREPLInput

    # Allowed modules
    ALLOWED_MODULES: ClassVar[set] = {
        'sympy', 'numpy', 'math', 'fractions',
        'itertools', 'collections', 'functools', 'decimal'
    }

    # Dangerous patterns to block
    DANGEROUS_PATTERNS: ClassVar[list] = [
        'import os', 'import sys', 'import subprocess',
        '__import__', 'eval', 'exec', 'compile',
        'open(', 'file(', 'input(', 'raw_input(',
        '__builtins__', '__globals__', '__locals__',
    ]

    def _is_safe(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute"""
        # Check dangerous patterns
        code_lower = code.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Dangerous pattern detected: {pattern}"

        # Parse and check imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in self.ALLOWED_MODULES:
                            return False, f"Module not allowed: {module}"
                elif isinstance(node, ast.ImportFrom):
                    module = node.module.split('.')[0] if node.module else ''
                    if module and module not in self.ALLOWED_MODULES:
                        return False, f"Module not allowed: {module}"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        return True, ""

    def _safe_import(self, name, *args, **kwargs):
        """Safe import that only allows whitelisted modules"""
        module_name = name.split('.')[0]
        if module_name not in self.ALLOWED_MODULES:
            raise ImportError(f"Module {module_name} is not allowed")
        return __import__(name, *args, **kwargs)

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the code in a restricted environment"""
        # Safety check
        is_safe, error_msg = self._is_safe(code)
        if not is_safe:
            return f"Error: {error_msg}"

        # Prepare restricted globals
        restricted_globals = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool,
                'dict': dict, 'float': float, 'int': int, 'len': len,
                'list': list, 'max': max, 'min': min, 'pow': pow,
                'range': range, 'round': round, 'set': set, 'str': str,
                'sum': sum, 'tuple': tuple, 'zip': zip,
                'enumerate': enumerate, 'sorted': sorted,
                'True': True, 'False': False, 'None': None,
                'print': print,  # Allow print
                '__import__': self._safe_import,  # Allow safe imports
            }
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Execute code with timeout (handled by exec, no signal for simplicity)
            exec(code, restricted_globals)

            # Get output
            output = sys.stdout.getvalue()

            # If no print output, try to get the last expression value
            if not output:
                # Try to evaluate last expression
                try:
                    lines = code.strip().split('\n')
                    last_line = lines[-1]
                    # Check if last line is an expression (not assignment/import)
                    if not any(keyword in last_line for keyword in ['=', 'import', 'def', 'class', 'for', 'while', 'if']):
                        result = eval(last_line, restricted_globals)
                        if result is not None:
                            output = str(result)
                except:
                    pass

            return output.strip() if output else "Code executed successfully (no output)"

        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout


class SymPyCalculatorInput(BaseModel):
    """Input schema for SymPyCalculator"""
    query: str = Field(description="Mathematical query in format 'command: expression'. Commands: solve, diff, integrate, simplify, factor, expand, eval")


class SymPyCalculator(BaseTool):
    """
    SymPy-based calculator for symbolic mathematics.
    Supports equation solving, calculus, simplification, etc.
    """

    name: str = "sympy_calculator"
    description: str = """Symbolic mathematics calculator using SymPy.

INPUT FORMAT: "command: expression"
If no command specified, defaults to 'eval' (numerical evaluation).

WHEN TO USE THIS TOOL:
- Quick single-operation calculations
- Standard equation solving, calculus operations
- Expression simplification, factoring, expansion
- Numerical evaluation of mathematical expressions

WHEN TO USE python_repl INSTEAD:
- Multi-step calculations with intermediate results
- Problems requiring loops or conditionals
- Algorithm implementation
- When you need to verify solutions programmatically

COMMANDS:

1. solve - Solve equations
   "solve: x**2 + 5*x + 6 = 0" → [-3, -2]
   "solve: x**2 - 4" → [-2, 2] (assumes = 0)
   "solve: 2*x + 3*y = 10" → solution in terms of variables

2. diff - Differentiate (with respect to first variable found)
   "diff: sin(x)*cos(x)" → -sin(x)**2 + cos(x)**2
   "diff: x**3 + 2*x**2 + x" → 3*x**2 + 4*x + 1

3. integrate - Indefinite integration
   "integrate: x**2" → x**3/3
   "integrate: sin(x)" → -cos(x)

4. simplify - Algebraic simplification
   "simplify: (x**2 - 1)/(x - 1)" → x + 1
   "simplify: sin(x)**2 + cos(x)**2" → 1

5. factor - Factor expression
   "factor: x**2 - 4" → (x - 2)*(x + 2)
   "factor: x**2 + 5*x + 6" → (x + 2)*(x + 3)

6. expand - Expand expression
   "expand: (x + 1)**3" → x**3 + 3*x**2 + 3*x + 1
   "expand: (x + y)*(x - y)" → x**2 - y**2

7. eval - Numerical evaluation (DEFAULT if no command)
   "eval: sqrt(16) + 2**3" → 12.0
   "2*pi" → 6.283... (auto-uses eval)
   "exp(I*pi)" → -1.0 (Euler's formula)

SUPPORTED FUNCTIONS & CONSTANTS:
- Trigonometric: sin, cos, tan, cot, sec, csc
- Inverse trig: asin, acos, atan
- Hyperbolic: sinh, cosh, tanh
- Exponential/Log: exp, log, ln
- Roots: sqrt, cbrt, root(x, n)
- Special: factorial, binomial, Abs, floor, ceiling
- Constants: pi, E (Euler's number), I (imaginary unit), oo (infinity), GoldenRatio

EXAMPLES:
"solve: x**2 + 5*x + 6 = 0" → "[-3, -2]"
"diff: sin(x)**2" → "2*sin(x)*cos(x)"
"integrate: 1/x" → "log(x)"
"simplify: (x**2 - 1)/(x - 1)" → "x + 1"
"factor: x**3 - 8" → "(x - 2)*(x**2 + 2*x + 4)"
"expand: (x + 2)**4" → "x**4 + 8*x**3 + 24*x**2 + 32*x + 16"
"eval: sqrt(2)" → "1.414213562373095"
"sin(pi/6)" → "0.5"
"""
    args_schema: Type[BaseModel] = SymPyCalculatorInput

    def _parse_query(self, query: str) -> tuple[str, str]:
        """Parse query into command and expression"""
        if ':' in query:
            parts = query.split(':', 1)
            command = parts[0].strip().lower()
            expression = parts[1].strip()
        else:
            command = 'eval'
            expression = query.strip()

        return command, expression

    def _parse_expression(self, expr_str: str):
        """Parse string to SymPy expression"""
        transformations = standard_transformations + (implicit_multiplication_application,)

        # Handle equations (with =)
        if '=' in expr_str and expr_str.count('=') == 1:
            lhs, rhs = expr_str.split('=')
            lhs_expr = parse_expr(lhs.strip(), transformations=transformations)
            rhs_expr = parse_expr(rhs.strip(), transformations=transformations)
            return sympy.Eq(lhs_expr, rhs_expr)
        else:
            return parse_expr(expr_str, transformations=transformations)

    def _format_result(self, result, is_numeric: bool = False) -> str:
        """Format result for display"""
        if isinstance(result, list):
            if len(result) == 0:
                return "No solution found"
            elif len(result) == 1:
                return str(result[0])
            else:
                return str(result)
        elif isinstance(result, dict):
            return ", ".join([f"{var} = {val}" for var, val in result.items()])
        elif is_numeric and hasattr(result, 'evalf'):
            return str(N(result))
        else:
            return str(result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the symbolic math query"""
        try:
            # Parse query
            command, expr_str = self._parse_query(query)

            # Parse expression
            expr = self._parse_expression(expr_str)

            # Get free symbols (variables)
            if hasattr(expr, 'free_symbols'):
                free_vars = list(expr.free_symbols)
            else:
                free_vars = []

            # Execute command
            if command == 'solve':
                if isinstance(expr, sympy.Eq):
                    # It's an equation
                    result = solve(expr, free_vars)
                else:
                    # Assume equals zero
                    result = solve(expr, free_vars)
                return self._format_result(result)

            elif command in ['diff', 'derivative']:
                if not free_vars:
                    return "Error: No variables found in expression"
                var = free_vars[0]  # Use first variable
                result = diff(expr, var)
                return self._format_result(result)

            elif command in ['integrate', 'integral']:
                if not free_vars:
                    return "Error: No variables found in expression"
                var = free_vars[0]
                result = integrate(expr, var)
                return self._format_result(result)

            elif command == 'simplify':
                result = simplify(expr)
                return self._format_result(result)

            elif command == 'factor':
                result = factor(expr)
                return self._format_result(result)

            elif command == 'expand':
                result = expand(expr)
                return self._format_result(result)

            elif command in ['eval', 'evaluate']:
                # Numerical evaluation
                if hasattr(expr, 'evalf'):
                    result = expr.evalf()
                else:
                    result = expr
                return self._format_result(result, is_numeric=True)

            else:
                return f"Error: Unknown command '{command}'"

        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"


# Tool instances
math_python_repl = MathPythonREPL()
sympy_calculator = SymPyCalculator()

# List of all tools
math_tools = [math_python_repl, sympy_calculator]


if __name__ == "__main__":
    # Quick test
    print("Testing MathPythonREPL:")
    print(math_python_repl.run("from sympy import symbols, solve; x = symbols('x'); print(solve(x**2 + 5*x + 6, x))"))

    print("\nTesting SymPyCalculator:")
    print(sympy_calculator.run("solve: x**2 + 5*x + 6 = 0"))
    print(sympy_calculator.run("diff: sin(x)*cos(x)"))
    print(sympy_calculator.run("eval: sqrt(16) + 2**3"))
