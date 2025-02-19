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

# We use the dimension vector order: [L, M, T, I, Θ, N, J].
#   L = length        (m)
#   M = mass          (kg)
#   T = time          (s)
#   I = current       (A)
#   Θ = temperature   (K)
#   N = amount        (mol)
#   J = luminous int. (cd)

BASE_DIMENSIONS = {
    "m":   [1, 0, 0, 0, 0, 0, 0],  # length
    "kg":  [0, 1, 0, 0, 0, 0, 0],  # mass
    "s":   [0, 0, 1, 0, 0, 0, 0],  # time
    "A":   [0, 0, 0, 1, 0, 0, 0],  # electric current
    "K":   [0, 0, 0, 0, 1, 0, 0],  # temperature
    "mol": [0, 0, 0, 0, 0, 1, 0],  # amount of substance
    "cd":  [0, 0, 0, 0, 0, 0, 1],  # luminous intensity
}

# A lookup for base-unit index -> symbol, for leftover exponents
INDEX_TO_BASE = ["m", "kg", "s", "A", "K", "mol", "cd"]


def zero_dim():
    return [0, 0, 0, 0, 0, 0, 0]

def dim_add(d1, d2):
    return [a + b for (a, b) in zip(d1, d2)]

def dim_sub(d1, d2):
    return [a - b for (a, b) in zip(d1, d2)]

def dim_mul(d, n):
    return [x * n for x in d]

def dim_eq(d1, d2):
    return all(a == b for a, b in zip(d1, d2))

def is_dimless(d):
    return all(x == 0 for x in d)


##############################################
# 2. COMMON DERIVED UNITS
##############################################
# We store each standard derived unit's 7D vector in the same [L, M, T, I, Θ, N, J] order.

DERIVED_UNITS = {
    (0, 0, -1, 0, 0, 0, 0): "Hz",     # 1/s
    (1, 1, -2, 0, 0, 0, 0): "N",      # kg*m/s^2
    (-1, 1, -2, 0, 0, 0, 0): "Pa",    # N/m^2
    (2, 1, -2, 0, 0, 0, 0): "J",      # N*m
    (2, 1, -3, 0, 0, 0, 0): "W",      # J/s
    (0, 0, 1, 1, 0, 0, 0): "C",       # A*s
    (2, 1, -3, -1, 0, 0, 0): "V",     # W/A = J/(C)
    (-2, -1, 4, 2, 0, 0, 0): "F",     # C/V
    (2, 1, -3, -2, 0, 0, 0): "Ω",     # V/A
    (-2, -1, 3, 2, 0, 0, 0): "S",     # 1/Ω
    (2, 1, -2, -1, 0, 0, 0): "Wb",    # V*s
    (0, 1, -2, -1, 0, 0, 0): "T",     # Wb/m^2
    (2, 1, -2, -2, 0, 0, 0): "H",     # Wb/A
    (0, 0, -1, 0, 0, 0, 0): "Bq",     # 1/s (same as Hz, but used for radioactivity)
    (2, 0, -2, 0, 0, 0, 0): "Gy",     # J/kg (also "Sv")
    (0, 0, -1, 0, 0, 1, 0): "kat",    # mol/s
    # ... add more if desired ...
}


##############################################
# 3. AST NODE DEFINITIONS
##############################################

class NumberNode:
    """
    Numeric literal (string) + optional raw unit token [ ... ].
    Example: value_str="1.5", unit_str="[m/s^2]"
    """
    def __init__(self, value_str, unit_str=None):
        self.value_str = value_str
        self.unit_str = unit_str

    def __repr__(self):
        if self.unit_str:
            return f"NumberNode(value={self.value_str}, unit={self.unit_str})"
        return f"NumberNode(value={self.value_str})"


class BinOpNode:
    """
    Binary operation node: left op right
    op in { '+', '-', '*', '/', '^' }
    """
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
    """
    Splits the input into tokens:
      - numeric tokens (possibly signed)
      - parentheses: ( )
      - operators: + - * / ^
      - bracketed unit chunk: [ ... ]
      - ignore whitespace
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if c.isspace():
            i += 1
            continue

        # Single-char operators or parentheses
        if c in ('(', ')', '+', '*', '/', '^'):
            tokens.append(c)
            i += 1
            continue

        # Distinguish minus as unary sign vs operator
        if c == '-':
            if (i + 1 < n) and (text[i+1].isdigit() or text[i+1] == '.'):
                # negative number
                start = i
                i += 1
                while i < n and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                tokens.append(text[start:i])
            else:
                # standalone operator
                tokens.append('-')
                i += 1
            continue

        # Bracketed unit: read until matching ']'
        if c == '[':
            start = i
            bracket_depth = 1
            i += 1
            while i < n and bracket_depth > 0:
                if text[i] == ']':
                    bracket_depth -= 1
                else:
                    i += 1
            if bracket_depth != 0:
                raise ValueError("Mismatched brackets '[' ']' in unit.")
            i += 1  # consume the ']'
            tokens.append(text[start:i])  # entire chunk: "[ ... ]"
            continue

        # Numbers
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
    """
    Grammar:
      Expr   -> Term (('+'|'-') Term)*
      Term   -> Factor (('*'|'/') Factor)*
      Factor -> Power ('^' Factor)?
      Power  -> NumberWithOptionalUnit | '(' Expr ')'
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

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
        """
        Factor -> Power ('^' Factor)?
        """
        base = self.parse_power()
        if self.match('^'):
            exponent_node = self.parse_factor()
            return BinOpNode(base, '^', exponent_node)
        return base

    def parse_power(self):
        """
        Power -> NumberWithOptionalUnit | '(' Expr ')'
        """
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

        # Must be numeric
        try:
            float(tok)
        except ValueError:
            raise ValueError(f"Expected numeric, got {tok}")

        self.pos += 1  # consume number
        unit_tok = None
        # If next is bracketed unit
        if self.current_token() and self.current_token().startswith('['):
            unit_tok = self.current_token()
            self.pos += 1

        return NumberNode(tok, unit_tok)


##############################################
# 6. INTERPRETER (DIMENSIONAL ANALYSIS)
##############################################

def evaluate(node):
    """
    Evaluate the AST node -> (value: float, dimension: [7]).
    """
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
            # dimension must match
            if not dim_eq(left_dim, right_dim):
                raise ValueError(
                    f"Dimension mismatch in addition: {left_dim} vs {right_dim}"
                )
            return (left_val + right_val, left_dim)

        elif op == '-':
            # dimension must match
            if not dim_eq(left_dim, right_dim):
                raise ValueError(
                    f"Dimension mismatch in subtraction: {left_dim} vs {right_dim}"
                )
            return (left_val - right_val, left_dim)

        elif op == '*':
            return (left_val * right_val, dim_add(left_dim, right_dim))

        elif op == '/':
            if right_val == 0:
                raise ValueError("Division by zero in numeric part.")
            return (left_val / right_val, dim_sub(left_dim, right_dim))

        elif op == '^':
            # exponent must be dimensionless
            if not dim_eq(right_dim, zero_dim()):
                raise ValueError("Exponent must be dimensionless.")
            exponent_int = int(right_val)  # typical usage integer
            return (left_val ** exponent_int, dim_mul(left_dim, exponent_int))

        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        raise TypeError("Invalid AST node type.")


def parse_unit_string(unit_tok):
    """
    e.g. "[m/s^2]" -> parse inside "m/s^2" => dimension vector
         "[-]" or empty => dimensionless
    """
    if not (unit_tok.startswith('[') and unit_tok.endswith(']')):
        raise ValueError(f"Invalid unit token: {unit_tok}")

    inside = unit_tok[1:-1].strip()  # remove brackets

    # allow "[-]" or empty to mean dimensionless
    if inside == '' or inside == '-':
        return zero_dim()

    # split on '*' or '/'
    parts = re.split(r'([\*/])', inside)
    parts = [p.strip() for p in parts if p.strip()]

    dim = zero_dim()
    current_op = '*'

    for part in parts:
        if part in ('*', '/'):
            current_op = part
        else:
            # part might be "m", "s^2", "kg^3", etc.
            base, exp = parse_base_exponent(part)
            if base not in BASE_DIMENSIONS:
                raise ValueError(
                    f"Unknown base unit '{base}' in unit string '{unit_tok}'. "
                    f"Allowed: {list(BASE_DIMENSIONS.keys())} or [-]."
                )
            factor_dim = dim_mul(BASE_DIMENSIONS[base], exp)
            if current_op == '*':
                dim = dim_add(dim, factor_dim)
            else:
                dim = dim_sub(dim, factor_dim)

    return dim

def parse_base_exponent(token):
    """
    e.g. "m^2" -> base=m, exponent=2
         "kg^3" -> base=kg, exponent=3
         "s" -> base=s, exponent=1
    """
    if '^' in token:
        base, exp_str = token.split('^', 1)
        e = int(exp_str.strip())
    else:
        base, e = token, 1
    return base.strip(), e


##############################################
# 7. UNIT SIMPLIFICATION LOGIC
##############################################

def can_factor_out(dim, derived):
    """
    Returns True if `dim - derived` does not cause any exponent to go below that in `dim`
    in a "bad" way. In other words, for each i, dim[i] >= derived[i].
    This naive approach only checks if all subtractions remain >= 0, 
    which is appropriate if we see each derived unit as an 'add' to exponents
    from zero. For negative exponents or more complex combos, one might generalize further.
    
    We'll do a simpler check: For us to "factor out" the derived vector dV from dim,
    we need: dim[i] - dV[i] >= 0 if dV[i] >= 0, 
    or dim[i] - dV[i] <= 0 if dV[i] < 0, etc.
    
    Actually it's easier: we want to see if we can subtract the derived vector 
    in full. That means dim[i] >= derived[i] for every i. (Because the exponents
    are added for multiplication. So to factor out, we subtract.)
    """
    for i in range(7):
        if dim[i] < derived[i]:
            return False
    return True


def factor_out_known_units(dim):
    """
    Tries to factor out known derived units (like N, J, Pa, etc.) from the dimension vector.
    Returns a list of (unit_name, count) plus the leftover dimension.

    Example:
      If dim == (1,1,-2,0,0,0,0), that matches "N" exactly => [("N",1)] leftover (0,0,0,0,0,0,0).
      If dim == (2,1,-3,0,0,0,0),
         we might factor out "N" once => leftover (1,0,-1,0,0,0,0) => "m/s", etc.
         We'll do a naive approach in a fixed order.

    This function does a "greedy" approach in the order we define below.
    """
    # We'll define a stable order to try factoring out.
    # Typically you might want to put bigger/ more "common" units first,
    # or you can do alphabetical. We'll just define an array from the dictionary.
    derived_list = list(DERIVED_UNITS.items())
    # E.g. [((0,0,-1,0,0,0,0),"Hz"), ((1,1,-2,0,0,0,0),"N"), ...]
    # Sort them in some consistent manner (not strictly necessary):
    derived_list.sort(key=lambda x: x[1])  # sort by name

    factors = []
    dim_left = dim[:]

    for vec, name in derived_list:
        count = 0
        # While we can factor out this derived unit, do so
        while can_factor_out(dim_left, vec):
            # subtract
            dim_left = dim_sub(dim_left, vec)
            count += 1
        if count > 0:
            factors.append((name, count))

    return factors, dim_left


def leftover_dim_to_string(dim):
    """
    Convert any leftover dimension exponents to a base-unit expression (m^x * kg^y / s^z, etc.).
    We'll do a standard numerator/denominator approach, ignoring standard derived unit names here.
    If dimensionless, return "" or "-".
    """
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
        # purely denominator
        return "1/" + "*".join(denominator_parts)
    elif not denominator_parts:
        return "*".join(numerator_parts)
    else:
        return "*".join(numerator_parts) + " / " + "*".join(denominator_parts)


def simplify_dimension(dim):
    """
    Attempt to factor out known derived units repeatedly, then represent leftover in base units.
    Produce a string like "N*m/s^2" or "Pa" or "Hz", etc.
    """
    if is_dimless(dim):
        return "-"  # or just ""

    # 1) Factor out known derived units
    factors, leftover = factor_out_known_units(dim)

    # 2) Build a string from those factors
    factor_strs = []
    for (unit_name, cnt) in factors:
        if cnt == 1:
            factor_strs.append(unit_name)
        else:
            # e.g. "N^2"
            factor_strs.append(f"{unit_name}^{cnt}")

    # 3) Convert leftover base exponents to a string
    left_str = leftover_dim_to_string(leftover)
    if left_str:
        factor_strs.append(left_str)

    # Combine them with '*' in between
    if not factor_strs:
        return "-"  # dimensionless after factoring
    else:
        # e.g. "N*m/s^2", "Wb/(mol*K^2)", etc.
        return "*".join(factor_strs)


##############################################
# 8. REPL & CLI
##############################################

console = Console()

def evaluate_expression(expr: str):
    """
    Parse & evaluate a single expression, returning (value, dimension) or raising an error.
    """
    tokens = tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse()
    return evaluate(ast)

def repl():
    """
    REPL using Rich. Enter expressions; 'q'/'quit'/'exit' to quit.
    """
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
            # Convert dimension to a simplified derived-unit string
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
        # REPL mode
        repl()
    else:
        # Evaluate each expression once
        for expr in args.expressions:
            try:
                val, dim = evaluate_expression(expr)
                dim_str = simplify_dimension(dim)
                console.print(f"[bold]{expr}[/] => [yellow]{val}[/] [magenta]{dim_str}[/]")
            except Exception as e:
                console.print(f"[red]ERROR in '{expr}':[/] {e}")

if __name__ == "__main__":
    main()
