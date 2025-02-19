# Copyright (c) 2025, Spaghetti Software Inc
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# This is a simple dimensional analysis calculator that can be used to evaluate expressions with 
# units. It supports basic arithmetic operations and variable assignments.
#
# Example usage:
# $ python3 oneunite.py
# This is oneunit v1.0: Dimensional Analysis REPL
# You can define variables, e.g. mass = 2 [kg]
# Then use them, e.g. mass * 9.8 [m/s^2]
# Type 'q', 'quit', or 'exit' to quit.
# dim> mass = 2 [kg]
# Result: 2.0 kg
# dim> force = mass * 9.8 [m/s^2]
# Result: 19.6 N
#


#!/usr/bin/env python3

import sys
import re
import argparse
from rich.console import Console

# Optionally import readline for arrow-key history on Unix-like systems
try:
    import readline
except ImportError:
    pass

##############################################
# 1. DIMENSIONS & BASE/DERIVED UNITS
##############################################

# Vector order: [L, M, T, I, Θ, N, J]
#  L=length (m), M=mass (kg), T=time (s),
#  I=current (A), Θ=temperature (K), N=amount (mol), J=luminous intensity (cd)

BASE_DIMENSIONS = {
    "m":   [1, 0, 0, 0, 0, 0, 0],
    "kg":  [0, 1, 0, 0, 0, 0, 0],
    "s":   [0, 0, 1, 0, 0, 0, 0],
    "A":   [0, 0, 0, 1, 0, 0, 0],
    "K":   [0, 0, 0, 0, 1, 0, 0],
    "mol": [0, 0, 0, 0, 0, 1, 0],
    "cd":  [0, 0, 0, 0, 0, 0, 1],
}

DERIVED_DIMENSIONS = {
    "Hz":  [0, 0, -1, 0, 0, 0, 0],
    "N":   [1, 1, -2, 0, 0, 0, 0],
    "Pa":  [-1, 1, -2, 0, 0, 0, 0],
    "J":   [2, 1, -2, 0, 0, 0, 0],
    "W":   [2, 1, -3, 0, 0, 0, 0],
    "C":   [0, 0, 1, 1, 0, 0, 0],
    "V":   [2, 1, -3, -1, 0, 0, 0],
    "F":   [-2, -1, 4, 2, 0, 0, 0],
    "Ohm":   [2, 1, -3, -2, 0, 0, 0],
    "S":   [-2, -1, 3, 2, 0, 0, 0],
    "Wb":  [2, 1, -2, -1, 0, 0, 0],
    "T":   [0, 1, -2, -1, 0, 0, 0],
    "H":   [2, 1, -2, -2, 0, 0, 0],
    "Bq":  [0, 0, -1, 0, 0, 0, 0],
    "Gy":  [2, 0, -2, 0, 0, 0, 0],
    "Sv":  [2, 0, -2, 0, 0, 0, 0],
    "kat": [0, 0, -1, 0, 0, 1, 0],
}

ALL_UNIT_DIMENSIONS = {}
ALL_UNIT_DIMENSIONS.update(BASE_DIMENSIONS)
ALL_UNIT_DIMENSIONS.update(DERIVED_DIMENSIONS)

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
# 2. SYMBOL TABLE FOR VARIABLES
##############################################

symbol_table = {}  
# Maps var_name -> (value: float, dimension: [7])

##############################################
# 3. AST NODE CLASSES
##############################################

class Node:
    pass

class NumberNode(Node):
    def __init__(self, value_str, unit_str=None):
        self.value_str = value_str
        self.unit_str = unit_str

    def __repr__(self):
        if self.unit_str:
            return f"NumberNode(value={self.value_str}, unit={self.unit_str})"
        else:
            return f"NumberNode(value={self.value_str})"

class VariableNode(Node):
    def __init__(self, var_name):
        self.var_name = var_name

    def __repr__(self):
        return f"VariableNode({self.var_name})"

class BinOpNode(Node):
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

        if c in ('(', ')', '+', '*', '/', '^'):
            tokens.append(c)
            i += 1
            continue

        if c == '-':
            if (i+1 < n) and (text[i+1].isdigit() or text[i+1] == '.'):
                start = i
                i += 1
                while i < n and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                tokens.append(text[start:i])
            else:
                tokens.append('-')
                i += 1
            continue

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
                raise ValueError("Mismatched '[' ']' in unit specification.")
            i += 1
            tokens.append(text[start:i])
            continue

        if c.isdigit() or c == '.':
            start = i
            i += 1
            while i < n and (text[i].isdigit() or text[i] == '.'):
                i += 1
            tokens.append(text[start:i])
            continue

        # otherwise treat as variable
        if c.isalpha():
            start = i
            i += 1
            while i < n and (text[i].isalnum() or text[i] == '_'):
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
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        self.pos += 1

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
            raise ValueError(
                f"Extra tokens after valid expression: {self.current_token()}"
            )
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
        node = self.parse_power()
        if self.match('^'):
            exponent_part = self.parse_factor()
            node = BinOpNode(node, '^', exponent_part)
        return node

    def parse_power(self):
        tok = self.current_token()
        if tok == '(':
            self.advance()
            node = self.parse_expr()
            self.expect(')')
            return node

        if self.is_numeric_token(tok):
            self.advance()
            unit_tok = None
            if self.current_token() and self.current_token().startswith('['):
                unit_tok = self.current_token()
                self.advance()
            return NumberNode(value_str=tok, unit_str=unit_tok)

        if tok is None:
            raise ValueError("Unexpected end of input in parse_power.")
        # assume variable
        self.advance()
        return VariableNode(tok)

    def is_numeric_token(self, tok):
        if tok is None:
            return False
        try:
            float(tok)
            return True
        except ValueError:
            return False

##############################################
# 6. EVALUATION
##############################################

def evaluate_ast(node):
    if isinstance(node, NumberNode):
        val = float(node.value_str)
        if node.unit_str:
            dim = parse_unit_string(node.unit_str)
        else:
            dim = zero_dim()
        return (val, dim)

    elif isinstance(node, VariableNode):
        var_name = node.var_name
        if var_name not in symbol_table:
            raise ValueError(f"Undefined variable '{var_name}'")
        return symbol_table[var_name]

    elif isinstance(node, BinOpNode):
        left_val, left_dim = evaluate_ast(node.left)
        right_val, right_dim = evaluate_ast(node.right)
        op = node.op

        if op == '+':
            if not dim_eq(left_dim, right_dim):
                raise ValueError(
                    f"Dimension mismatch in addition: {left_dim} vs {right_dim}"
                )
            return (left_val + right_val, left_dim)

        elif op == '-':
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
            if not dim_eq(right_dim, zero_dim()):
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

    parts = re.split(r'([\*/])', inside)
    parts = [p.strip() for p in parts if p.strip()]

    dim = zero_dim()
    current_op = '*'
    for part in parts:
        if part in ('*', '/'):
            current_op = part
        else:
            base, exponent = parse_base_exponent(part)
            if base not in ALL_UNIT_DIMENSIONS:
                raise ValueError(
                    f"Unknown base/derived unit '{base}' in '{unit_tok}'."
                )
            factor_dim = dim_mul(ALL_UNIT_DIMENSIONS[base], exponent)
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
# 7. PRINTING: DERIVED + BASE
##############################################

def dim_to_derived_plus_base_string(dim):
    """
    1) If dim is dimensionless, return '-'
    2) If dim matches a known derived unit exactly, return e.g. "N (m*kg / s^2)"
    3) Otherwise, just return base expansion (e.g. "m*kg / s^2")
    """
    if is_dimless(dim):
        return "-"

    # Check for an exact match in DERIVED_DIMENSIONS
    for name, vector in DERIVED_DIMENSIONS.items():
        if vector == dim:
            base_str = dim_to_base_string(dim)
            # If the base_str is just "-", that means dimensionless. Usually won't happen for derived though.
            if base_str == "-":
                return name
            # otherwise "N (kg*m / s^2)", etc.
            return f"{name} ({base_str})"

    # fallback: no derived name recognized
    return dim_to_base_string(dim)

def dim_to_base_string(dim):
    """
    Example: [1,1,-2,0,0,0,0] => 'm*kg / s^2'
    """
    numerator_parts = []
    denominator_parts = []
    for i, exponent in enumerate(dim):
        if exponent == 0:
            continue
        base_symbol = INDEX_TO_BASE[i]
        exp_abs = abs(exponent)
        if exp_abs == 1:
            txt = base_symbol
        else:
            txt = f"{base_symbol}^{exp_abs}"

        if exponent > 0:
            numerator_parts.append(txt)
        else:
            denominator_parts.append(txt)

    if not numerator_parts and not denominator_parts:
        return "-"

    if not numerator_parts:
        return "1/" + "*".join(denominator_parts)
    elif not denominator_parts:
        return "*".join(numerator_parts)
    else:
        return "*".join(numerator_parts) + " / " + "*".join(denominator_parts)

##############################################
# 8. ERROR REPORTING
##############################################

def report_parse_error(expr, tokens, parser, exc):
    msg = []
    msg.append(f"ERROR parsing expression: {expr}")
    msg.append(f"Tokens: {tokens}")
    msg.append(f"Parser position: {parser.pos}")
    msg.append(f"Details: {exc}")
    return "\n".join(msg)

##############################################
# 9. MAIN LOGIC (REPL, FILE, CLI)
##############################################

console = Console()

def evaluate_expression(expr):
    if '=' in expr:
        lhs, rhs = expr.split('=', 1)
        var_name = lhs.strip()
        if not var_name:
            raise ValueError("Invalid assignment (no variable name).")
        (val, dim) = parse_and_evaluate(rhs.strip())
        symbol_table[var_name] = (val, dim)
        return (val, dim)
    else:
        return parse_and_evaluate(expr)

def parse_and_evaluate(expr):
    tokens = tokenize(expr)
    parser = Parser(tokens)
    try:
        ast = parser.parse()
    except Exception as e:
        raise ValueError(report_parse_error(expr, tokens, parser, e))
    return evaluate_ast(ast)

def repl():
    console.print("[bold cyan]This is oneunit v1.0: Dimensional Analysis REPL[/]")
    console.print("You can define variables, e.g. mass = 2 [kg]")
    console.print("Then use them, e.g. mass * 9.8 [m/s^2]")
    console.print("Type 'q', 'quit', or 'exit' to quit.\n")

    while True:
        expr = console.input("[bold green]dim> [/]")
        if expr.lower() in ("q", "quit", "exit"):
            console.print("Goodbye!")
            break

        expr = expr.strip()
        if not expr:
            continue

        try:
            val, dim = evaluate_expression(expr)
            dim_str = dim_to_derived_plus_base_string(dim)
            console.print(f"[yellow]Result:[/] {val} [magenta]{dim_str}[/]")
        except Exception as e:
            console.print(f"[red]{e}[/]")

def process_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                val, dim = evaluate_expression(line)
                dim_str = dim_to_derived_plus_base_string(dim)
                console.print(f"[bold]{line}[/] => [yellow]{val}[/] [magenta]{dim_str}[/]")
            except Exception as e:
                console.print(f"[red]Error in line: '{line}'[/]")
                console.print(f"[red]{e}[/]")

def main():
    ap = argparse.ArgumentParser(description="Dimensional Analysis Calculator")
    ap.add_argument("-f", "--file", help="File containing expressions, one per line")
    ap.add_argument("expressions", nargs="*", help="Expressions to evaluate")
    args = ap.parse_args()

    if args.file:
        process_file(args.file)

    for expr in args.expressions:
        try:
            val, dim = evaluate_expression(expr)
            dim_str = dim_to_derived_plus_base_string(dim)
            console.print(f"[bold]{expr}[/] => [yellow]{val}[/] [magenta]{dim_str}[/]")
        except Exception as e:
            console.print(f"[red]Error in expression '{expr}':[/]")
            console.print(f"[red]{e}[/]")

    if not args.file and not args.expressions:
        repl()

if __name__ == "__main__":
    main()
