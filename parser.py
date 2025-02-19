#!/usr/bin/env python3

import sys
import re
import argparse

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

##############################################
# 1. DIMENSION CONSTANTS & HELPERS
##############################################

# We use dimension vector order: [L, M, T, I, Θ, N, J].
BASE_DIMENSIONS = {
    "m":   [1, 0, 0, 0, 0, 0, 0],
    "kg":  [0, 1, 0, 0, 0, 0, 0],
    "s":   [0, 0, 1, 0, 0, 0, 0],
    "A":   [0, 0, 0, 1, 0, 0, 0],
    "K":   [0, 0, 0, 0, 1, 0, 0],
    "mol": [0, 0, 0, 0, 0, 1, 0],
    "cd":  [0, 0, 0, 0, 0, 0, 1],
}

INDEX_TO_BASE = ["m", "kg", "s", "A", "K", "mol", "cd"]

def zero_dim():
    return [0]*7

def dim_add(d1, d2):
    return [a + b for a, b in zip(d1, d2)]

def dim_sub(d1, d2):
    return [a - b for a, b in zip(d1, d2)]

def dim_mul(d, n):
    return [x * n for x in d]

def dim_eq(d1, d2):
    return all(a == b for a, b in zip(d1, d2))

def is_dimless(d):
    return all(x == 0 for x in d)

##############################################
# 2. COMMON DERIVED UNITS
##############################################

DERIVED_UNITS = {
    (0, 0, -1, 0, 0, 0, 0): "Hz",
    (1, 1, -2, 0, 0, 0, 0): "N",
    (-1, 1, -2, 0, 0, 0, 0): "Pa",
    (2, 1, -2, 0, 0, 0, 0): "J",
    (2, 1, -3, 0, 0, 0, 0): "W",
    (0, 0, 1, 1, 0, 0, 0): "C",
    (2, 1, -3, -1, 0, 0, 0): "V",
    (-2, -1, 4, 2, 0, 0, 0): "F",
    (2, 1, -3, -2, 0, 0, 0): "Ω",
    (-2, -1, 3, 2, 0, 0, 0): "S",
    (2, 1, -2, -1, 0, 0, 0): "Wb",
    (0, 1, -2, -1, 0, 0, 0): "T",
    (2, 1, -2, -2, 0, 0, 0): "H",
    (0, 0, -1, 0, 0, 0, 0): "Bq",   # same as Hz dimensionally
    (2, 0, -2, 0, 0, 0, 0): "Gy",   # same as Sv
    (0, 0, -1, 0, 0, 1, 0): "kat",
}


##############################################
# 3. AST NODES
##############################################

class NumberNode:
    def __init__(self, value_str, unit_str=None):
        self.value_str = value_str
        self.unit_str = unit_str

    def __repr__(self):
        if self.unit_str:
            return f"NumberNode(value={self.value_str}, unit={self.unit_str})"
        return f"NumberNode(value={self.value_str})"

class BinOpNode:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinOpNode({self.left} {self.op} {self.right})"

##############################################
# 4. LEXER
##############################################

def tokenize(text):
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if c.isspace():
            i += 1
            continue

        # Operators or parentheses
        if c in ('(', ')', '+', '*', '/', '^'):
            tokens.append(c)
            i += 1
            continue

        # Distinguish minus sign
        if c == '-':
            if (i + 1 < n) and (text[i+1].isdigit() or text[i+1] == '.'):
                start = i
                i += 1
                while i < n and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                tokens.append(text[start:i])
            else:
                tokens.append('-')
                i += 1
            continue

        # Bracketed unit
        if c == '[':
            start = i
            i += 1
            bracket_depth = 1
            while i < n and bracket_depth > 0:
                if text[i] == ']':
                    bracket_depth -= 1
                else:
                    i += 1
            if bracket_depth != 0:
                raise ValueError("Unmatched '[' ']' in unit.")
            i += 1
            tokens.append(text[start:i])
            continue

        # Number
        if c.isdigit() or c == '.':
            start = i
            i += 1
            while i < n and (text[i].isdigit() or text[i] == '.'):
                i += 1
            tokens.append(text[start:i])
            continue

        raise ValueError(f"Unexpected character '{c}' at index {i}")

    return tokens

##############################################
# 5. PARSER
##############################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def match(self, *expected):
        tok = self.current_token()
        if tok in expected:
            self.pos += 1
            return tok
        return None

    def expect(self, *expected):
        tok = self.current_token()
        if tok not in expected:
            raise ValueError(f"Expected one of {expected}, got {tok}")
        self.pos += 1
        return tok

    def parse(self):
        node = self.parse_expr()
        if self.current_token() is not None:
            raise ValueError(f"Extra tokens after valid expression: {self.current_token()}")
        return node

    def parse_expr(self):
        node = self.parse_term()
        while True:
            op = self.match('+', '-')
            if not op:
                break
            right = self.parse_term()
            node = BinOpNode(node, op, right)
        return node

    def parse_term(self):
        node = self.parse_factor()
        while True:
            op = self.match('*', '/')
            if not op:
                break
            right = self.parse_factor()
            node = BinOpNode(node, op, right)
        return node

    def parse_factor(self):
        base = self.parse_power()
        if self.match('^'):
            exponent_node = self.parse_factor()
            return BinOpNode(base, '^', exponent_node)
        return base

    def parse_power(self):
        if self.match('('):
            node = self.parse_expr()
            self.expect(')')
            return node
        else:
            return self.parse_number_with_unit()

    def parse_number_with_unit(self):
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of input (expected number).")

        # Must parse as float
        try:
            float(tok)
        except ValueError:
            raise ValueError(f"Expected numeric, got {tok}")
        self.pos += 1

        unit_tok = None
        if self.current_token() and self.current_token().startswith('['):
            unit_tok = self.current_token()
            self.pos += 1

        return NumberNode(tok, unit_tok)

##############################################
# 6. INTERPRETER
##############################################

def evaluate(node):
    if isinstance(node, NumberNode):
        val = float(node.value_str)
        if node.unit_str:
            dim = parse_unit_string(node.unit_str)
        else:
            dim = zero_dim()
        return (val, dim)

    elif isinstance(node, BinOpNode):
        left_val, left_dim = evaluate(node.left)
        right_val, right_dim = evaluate(node.right)
        op = node.op

        if op == '+':
            if not dim_eq(left_dim, right_dim):
                raise ValueError(f"Dimension mismatch in addition: {left_dim} vs {right_dim}")
            return (left_val + right_val, left_dim)

        elif op == '-':
            if not dim_eq(left_dim, right_dim):
                raise ValueError(f"Dimension mismatch in subtraction: {left_dim} vs {right_dim}")
            return (left_val - right_val, left_dim)

        elif op == '*':
            return (left_val * right_val, dim_add(left_dim, right_dim))

        elif op == '/':
            if right_val == 0:
                raise ValueError("Division by zero in numeric part.")
            return (left_val / right_val, dim_sub(left_dim, right_dim))

        elif op == '^':
            if not is_dimless(right_dim):
                raise ValueError("Exponent must be dimensionless.")
            exponent_int = int(right_val)
            return (left_val ** exponent_int, dim_mul(left_dim, exponent_int))

        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        raise TypeError("Invalid AST node.")

def parse_unit_string(unit_tok):
    if not (unit_tok.startswith('[') and unit_tok.endswith(']')):
        raise ValueError(f"Invalid unit token: {unit_tok}")
    inside = unit_tok[1:-1].strip()
    if inside == '' or inside == '-':
        return zero_dim()

    import re
    parts = re.split(r'([\*/])', inside)
    parts = [p.strip() for p in parts if p.strip()]

    dim = zero_dim()
    current_op = '*'
    for part in parts:
        if part in ('*', '/'):
            current_op = part
        else:
            base, exp = parse_base_exponent(part)
            if base not in BASE_DIMENSIONS:
                raise ValueError(
                    f"Unknown base unit '{base}' in '{unit_tok}'. "
                    f"Allowed: {list(BASE_DIMENSIONS.keys())} or [-]."
                )
            factor_dim = dim_mul(BASE_DIMENSIONS[base], exp)
            if current_op == '*':
                dim = dim_add(dim, factor_dim)
            else:
                dim = dim_sub(dim, factor_dim)

    return dim

def parse_base_exponent(token):
    if '^' in token:
        base, exp_str = token.split('^', 1)
        e = int(exp_str.strip())
    else:
        base, e = token, 1
    return base.strip(), e

##############################################
# 7. FACTORING OUT DERIVED UNITS
##############################################

def simplify_dimension(dim):
    if is_dimless(dim):
        return "-"

    # Factor out known derived units first
    factors, leftover = factor_out_known_units(dim)

    # Build factor string
    factor_strs = []
    for (unit_name, cnt) in factors:
        if cnt == 1:
            factor_strs.append(unit_name)
        else:
            factor_strs.append(f"{unit_name}^{cnt}")

    # Then leftover base exponents
    left_str = leftover_dim_to_string(leftover)
    if left_str:
        factor_strs.append(left_str)

    if not factor_strs:
        return "-"
    else:
        return "*".join(factor_strs)


def leftover_dim_to_string(dim):
    if is_dimless(dim):
        return ""
    numerator_parts = []
    denominator_parts = []
    for i, exponent in enumerate(dim):
        if exponent == 0:
            continue
        base_symbol = INDEX_TO_BASE[i]
        e_abs = abs(exponent)
        if e_abs == 1:
            exp_str = base_symbol
        else:
            exp_str = f"{base_symbol}^{e_abs}"
        if exponent > 0:
            numerator_parts.append(exp_str)
        else:
            denominator_parts.append(exp_str)

    if not numerator_parts and not denominator_parts:
        return ""

    if not numerator_parts:
        return "1/" + "*".join(denominator_parts)
    elif not denominator_parts:
        return "*".join(numerator_parts)
    else:
        return "*".join(numerator_parts) + " / " + "*".join(denominator_parts)


##
## CRITICAL FIX: Adjust "can_factor_out" so we DON'T infinitely factor out negative exponents.
##
def can_factor_out(dim, derived):
    """
    We only factor out the 'derived' vector if leftover[i] - derived[i] doesn't push
    any exponent in the wrong direction. The rule:
      if derived[i] > 0 => leftover[i] >= derived[i]
      if derived[i] < 0 => leftover[i] <= derived[i]
    i.e. we don't want to raise leftover exponents by subtracting negative exponents.
    """
    for i in range(7):
        needed = derived[i]
        have = dim[i]
        if needed > 0 and have < needed:
            return False
        if needed < 0 and have > needed:
            return False
    return True

def factor_out_known_units(dim):
    derived_list = list(DERIVED_UNITS.items())
    # sort by unit name to have a stable order
    derived_list.sort(key=lambda x: x[1])

    factors = []
    dim_left = dim[:]

    for vec, name in derived_list:
        count = 0
        # Repeatedly factor out if possible
        while can_factor_out(dim_left, vec):
            # subtract
            dim_left = dim_sub(dim_left, vec)
            count += 1
        if count > 0:
            factors.append((name, count))

    return factors, dim_left


##############################################
# 8. REPL & CLI
##############################################

console = Console()

def evaluate_expression(expr: str):
    tokens = tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse()
    return evaluate(ast)

def repl():
    console.print("[bold cyan]Welcome to the Dimensional Analysis REPL![/]")
    console.print("Enter an expression, or 'q'/'quit'/'exit' to quit.\n")

    while True:
        expr = Prompt.ask("[bold green]dim>[/]")
        if expr.lower() in ("q", "quit", "exit"):
            console.print("Goodbye!")
            break

        if not expr.strip():
            continue

        try:
            val, dim = evaluate_expression(expr)
            dim_str = simplify_dimension(dim)
            console.print(f"[yellow]Result:[/] {val} [magenta]{dim_str}[/]")
        except Exception as e:
            console.print(f"[red]ERROR:[/] {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Dimensional Analysis Calculator with optional REPL"
    )
    parser.add_argument(
        "expressions", nargs="*",
        help="Expressions to evaluate. If none provided, enters REPL mode."
    )
    args = parser.parse_args()

    if not args.expressions:
        repl()
    else:
        for expr in args.expressions:
            try:
                val, dim = evaluate_expression(expr)
                dim_str = simplify_dimension(dim)
                console.print(f"[bold]{expr}[/] => [yellow]{val}[/] [magenta]{dim_str}[/]")
            except Exception as e:
                console.print(f"[red]ERROR in '{expr}':[/] {e}")

if __name__ == "__main__":
    main()
