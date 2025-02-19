#!/usr/bin/env python3

import sys
import re
import argparse
from rich.console import Console
from rich.prompt import Prompt

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
    "Ω":   [2, 1, -3, -2, 0, 0, 0],
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
    """Represents a numeric literal with optional bracketed unit string."""
    def __init__(self, value_str, unit_str=None):
        self.value_str = value_str
        self.unit_str = unit_str

    def __repr__(self):
        if self.unit_str:
            return f"NumberNode(value={self.value_str}, unit={self.unit_str})"
        else:
            return f"NumberNode(value={self.value_str})"

class VariableNode(Node):
    """Represents a reference to a previously defined variable."""
    def __init__(self, var_name):
        self.var_name = var_name

    def __repr__(self):
        return f"VariableNode({self.var_name})"

class BinOpNode(Node):
    """Binary operation: left op right (e.g., +, -, *, /, ^)."""
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
      - variable names (alphabetic strings) outside brackets
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
            # If next char is digit or '.', parse a negative number
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
                raise ValueError("Mismatched '[' ']' in unit specification.")
            i += 1  # consume the ']'
            tokens.append(text[start:i])  # entire chunk: "[ ... ]"
            continue

        # Numbers (unsigned, if minus was handled above)
        if c.isdigit() or c == '.':
            start = i
            i += 1
            while i < n and (text[i].isdigit() or text[i] == '.'):
                i += 1
            tokens.append(text[start:i])
            continue

        # Otherwise, we assume it's a variable name (alphabetic)
        if c.isalpha():
            start = i
            i += 1
            # allow letters, digits, underscore in variable names, if desired:
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
    """
    Grammar:
      Expr   -> Term (('+'|'-') Term)*
      Term   -> Factor (('*'|'/') Factor)*
      Factor -> Power ('^' Factor)?
      Power  -> VariableRef | NumberWithOptionalUnit | '(' Expr ')'

    + We handle variable references if the token is purely alphabetic outside brackets.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

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
        """ Factor -> Power ('^' Factor)? """
        node = self.parse_power()
        if self.match('^'):
            exponent_part = self.parse_factor()
            node = BinOpNode(node, '^', exponent_part)
        return node

    def parse_power(self):
        """
        Power -> VariableRef | NumberWithOptionalUnit | '(' Expr ')'
        If current token is '(' => parse subexpr.
        If it's numeric => parse number+unit.
        If it's alphabetic => variable reference.
        """
        tok = self.current_token()
        if tok == '(':
            self.advance()  # consume '('
            node = self.parse_expr()
            self.expect(')')
            return node

        # Check if it's purely numeric -> parse a number+optional bracket
        if self.is_numeric_token(tok):
            self.advance()
            # maybe next token is bracket
            unit_tok = None
            if self.current_token() and self.current_token().startswith('['):
                unit_tok = self.current_token()
                self.advance()
            return NumberNode(value_str=tok, unit_str=unit_tok)

        # Otherwise, assume it's a variable name
        if tok is None:
            raise ValueError("Unexpected end of input in parse_power.")
        # We'll check that it's not an operator, bracket, etc.
        # We won't be super strict here: if it's not numeric or bracket, assume variable
        self.advance()  # consume the variable token
        return VariableNode(tok)

    def is_numeric_token(self, tok):
        if tok is None:
            return False
        # A token is numeric if it can convert to float
        try:
            float(tok)
            return True
        except ValueError:
            return False


##############################################
# 6. EVALUATION
##############################################

def evaluate_ast(node):
    """Recursively evaluate the AST node -> (value: float, dimension: [7])."""
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
        return symbol_table[var_name]  # (value, dim)

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
            exponent_int = int(right_val)  # typical usage integer
            return (left_val ** exponent_int, dim_mul(left_dim, exponent_int))

        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        raise TypeError("Invalid AST node type in evaluate_ast.")


def parse_unit_string(unit_tok):
    """
    e.g. "[m/s^2]", "[J/mol]", "[Hz]", "[kg*m^2/s^2]"...
    We'll parse them like a mini expression with '*' and '/'.
    """
    if not (unit_tok.startswith('[') and unit_tok.endswith(']')):
        raise ValueError(f"Invalid unit token: {unit_tok}")

    inside = unit_tok[1:-1].strip()

    # '[-]' or empty => dimensionless
    if inside == '' or inside == '-':
        return zero_dim()

    # parse by splitting on * and /
    parts = re.split(r'([\*/])', inside)
    parts = [p.strip() for p in parts if p.strip()]

    dim = zero_dim()
    current_op = '*'
    for part in parts:
        if part in ('*', '/'):
            current_op = part
        else:
            # possibly "kg", "m^2", "Hz", "J^3", etc.
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
    """
    e.g. "m^2" -> (m,2), "kg^3"->(kg,3), "Hz"->(Hz,1)
    """
    if '^' in token:
        base, exp_str = token.split('^', 1)
        e = int(exp_str.strip())
    else:
        base, e = token, 1
    return base.strip(), e


##############################################
# 7. BASIC DIM -> STRING
##############################################

def dim_to_basic_string(dim):
    """Show leftover exponents in base form, ignoring derived names."""
    if is_dimless(dim):
        return "-"

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
    """
    Provide a more detailed error message:
    - Show the expression,
    - Show tokens,
    - Show parser position,
    - Then show the exception message.
    """
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
    """
    Parse & evaluate a single expression or assignment (var=expr).
    If it's 'var = <something>', store the result in symbol_table.
    Else return (value, dimension).
    """
    # 1) Check if it's an assignment
    if '=' in expr:
        # e.g. "mass = 2 [kg]"
        lhs, rhs = expr.split('=', 1)
        var_name = lhs.strip()
        if not var_name:
            raise ValueError("Invalid assignment (no variable name).")
        # parse & evaluate RHS
        (val, dim) = parse_and_evaluate(rhs.strip())
        # store in symbol table
        symbol_table[var_name] = (val, dim)
        return (val, dim)  # or we can return None, but let's return the result
    else:
        return parse_and_evaluate(expr)

def parse_and_evaluate(expr):
    """Tokenize, parse, evaluate the expression => (value, dimension)."""
    tokens = tokenize(expr)
    parser = Parser(tokens)
    try:
        ast = parser.parse()
    except Exception as e:
        raise ValueError(report_parse_error(expr, tokens, parser, e))

    return evaluate_ast(ast)


def repl():
    console.print("[bold cyan]Welcome to the Dimensional Analysis REPL![/]")
    console.print("You can define variables: e.g. mass = 2 [kg]")
    console.print("Then use them: e.g. mass * 9.8 [m/s^2]")
    console.print("Type 'q', 'quit', or 'exit' to quit.\n")

    while True:
        expr = Prompt.ask("[bold green]dim>[/]")
        if expr.lower() in ("q", "quit", "exit"):
            console.print("Goodbye!")
            break

        expr = expr.strip()
        if not expr:
            continue

        try:
            val, dim = evaluate_expression(expr)
            dim_str = dim_to_basic_string(dim)
            console.print(f"[yellow]Result:[/] {val} [magenta]{dim_str}[/]")
        except Exception as e:
            console.print(f"[red]{e}[/]")


def process_file(filename):
    """Read expressions line by line from filename, ignoring empty or commented lines."""
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                val, dim = evaluate_expression(line)
                dim_str = dim_to_basic_string(dim)
                console.print(f"[bold]{line}[/] => [yellow]{val}[/] [magenta]{dim_str}[/]")
            except Exception as e:
                console.print(f"[red]Error in line: '{line}'[/]")
                console.print(f"[red]{e}[/]")


def main():
    ap = argparse.ArgumentParser(description="Dimensional Analysis Calculator")
    ap.add_argument("-f", "--file", help="File containing expressions, one per line")
    ap.add_argument("expressions", nargs="*", help="Expressions to evaluate")
    args = ap.parse_args()

    # 1) If a file is provided, process it first
    if args.file:
        process_file(args.file)

    # 2) Then process any expressions from the command line
    for expr in args.expressions:
        try:
            val, dim = evaluate_expression(expr)
            dim_str = dim_to_basic_string(dim)
            console.print(f"[bold]{expr}[/] => [yellow]{val}[/] [magenta]{dim_str}[/]")
        except Exception as e:
            console.print(f"[red]Error in expression '{expr}':[/]")
            console.print(f"[red]{e}[/]")

    # 3) If no file and no expressions, drop into REPL
    if not args.file and not args.expressions:
        repl()


if __name__ == "__main__":
    main()
