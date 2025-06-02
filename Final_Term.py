from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Callable, Set, FrozenSet
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from lark import Lark, Transformer, Token, Tree

# Define the semantic domains
type Graph = Set[frozenset]
type DVal = Graph

# Environment as a function
type Environment = Callable[[str], DVal]

def lookup(env: Environment, name: str) -> DVal:
    """Look up an identifier in the environment"""
    return env(name)

def bind(env: Environment, name: str, value: DVal) -> Environment:
    """Create new environment with an added binding"""
    def new_env(n: str) -> DVal:
        if n == name:
            return value
        return env(n)
    return new_env

# =====================================================================
# 1. Grammar for the extended DSL
# =====================================================================
# The grammar supports:
#   - command sequences (var declarations, assignments, print, if, while, function decl)
#   - expression grammar (let, logical, arithmetic, relational, graph operators, function calls)
grammar = r"""
program: command_seq

?command_seq: command (";" command_seq)?
?command: vardecl
        | assign
        | print_stmt
        | if_stmt
        | while_stmt
        | fun_decl

vardecl: "var" IDENTIFIER "=" expr      -> vardecl
assign: IDENTIFIER "<-" expr            -> assign
print_stmt: "print" expr               -> print_stmt
if_stmt: "if" expr "then" command_seq "else" command_seq "endif" -> if_stmt
while_stmt: "while" expr "do" command_seq "done" -> while_stmt
fun_decl: "function" IDENTIFIER "(" param_list? ")" "=" expr -> fun_decl

param_list: IDENTIFIER ("," IDENTIFIER)*
arg_list: expr ("," expr)*

funapp: IDENTIFIER "(" arg_list? ")"      -> funapp

?expr: let_expr
     | graph_expr

let_expr: "let" IDENTIFIER "=" expr "in" expr  -> let_expr

OP_G: "union" | "intersect" | "difference" | "path" | "shortest"
?graph_expr: graph_expr OP_G or_expr   -> graph_bin
           | or_expr

?or_expr: or_expr "or" and_expr   -> or_op
        | and_expr
?and_expr: and_expr "and" eq_expr -> and_op
         | eq_expr
?eq_expr: eq_expr "==" comp_expr  -> eq
        | eq_expr "!=" comp_expr -> neq
        | comp_expr
?comp_expr: comp_expr "<=" add_expr -> le
          | comp_expr ">=" add_expr -> ge
          | comp_expr "<" add_expr  -> lt
          | comp_expr ">" add_expr  -> gt
          | add_expr
?add_expr: add_expr "+" mul_expr    -> add
         | add_expr "-" mul_expr    -> sub
         | mul_expr
?mul_expr: mul_expr "*" atom        -> mul
         | mul_expr "/" atom        -> div
         | atom

?atom: NUMBER           -> number
     | "true"          -> true
     | "false"         -> false
     | funapp
     | IDENTIFIER      -> var
     | GRAPH           -> graphlit
     | NODES           -> nodes
     | "(" expr ")"

%import common.CNAME -> IDENTIFIER
%import common.INT -> NUMBER
%ignore /[ \t\f\r\n]+/  // skip whitespace
GRAPH: /'[^']*'/
NODES: "{" IDENTIFIER ("," IDENTIFIER)* "}"
"""
# Create the Lark parser for our grammar
parser = Lark(grammar, start='program')

# =====================================================================
# 2. AST (Abstract Syntax Tree) Node Definitions
# =====================================================================
# We define dataclasses for each kind of AST node (commands and expressions).

# -- Program and Commands --
@dataclass
class Program:
    commands: List  # list of Command nodes

@dataclass
class VarDecl:
    name: str
    expr: any       # Expression node

@dataclass
class Assign:
    name: str
    expr: any       # Expression node

@dataclass
class Print:
    expr: any       # Expression node

@dataclass
class If:
    cond: any            # Expression node
    then_branch: List    # list of Command nodes
    else_branch: List    # list of Command nodes

@dataclass
class While:
    cond: any            # Expression node
    body: List           # list of Command nodes

@dataclass
class FunctionDecl:
    name: str
    params: List[str]    # parameter names
    body: any            # Expression node

# -- Expressions --
@dataclass
class LetExpr:
    name: str
    expr: any        # Expression to bind
    body: any        # Expression in which 'name' is bound

@dataclass
class BinaryOp:
    op: str
    left: any        # left Expression
    right: any       # right Expression

@dataclass
class Number:
    value: int       # integer literal value

@dataclass
class Bool:
    value: bool      # boolean literal value

@dataclass
class Var:
    name: str        # variable reference

@dataclass
class GraphLiteral:
    filename: str    # graph file name (string)

@dataclass
class NodeSet:
    nodes: List[str] # node-list literal {A,B,C}

@dataclass
class FunctionCall:
    name: str
    args: List      # list of Expression arguments

# Closure class to capture function environment
@dataclass
class Closure:
    function: FunctionDecl  # function declaration AST
    env: List[dict]         # captured environment frames

# =====================================================================
# 3. Parse Tree to AST Transformer
# =====================================================================
# The Transformer will traverse the parse tree and build our AST nodes.
class ASTTransformer(Transformer):
    def program(self, args):
        # Args is result of command_seq
        return Program(args[0] if args else [])

    def command_seq(self, args):
        # Flatten a sequence of commands: return a flat list
        if not args:
            return []
        if len(args) == 1:
            # Single command (or list returned by recursion)
            first = args[0]
            if isinstance(first, list):
                return first
            else:
                return [first]
        # args[0] is first command, args[1] is list of remaining
        first = args[0]
        rest = args[1]
        if isinstance(rest, list):
            return [first] + rest
        else:
            return [first, rest]

    # Command transformers
    def vardecl(self, args):
        name = args[0].value
        return VarDecl(name, args[1])

    def assign(self, args):
        name = args[0].value
        return Assign(name, args[1])

    def print_stmt(self, args):
        return Print(args[0])

    def if_stmt(self, args):
        cond       = args[0]
        then_block = args[1]
        else_block = args[2]
    
        # se il ramo è un singolo comando, impacchettalo in lista
        if not isinstance(then_block, list):
            then_block = [then_block]
        if not isinstance(else_block, list):
            else_block = [else_block]
    
        return If(cond, then_block, else_block)

    def while_stmt(self, args):
        cond = args[0]
        body = args[1]
        if not isinstance(body, list):
            body = [body]
        return While(cond, body)

    def fun_decl(self, args):
        name = args[0].value
        if len(args) == 3:
            params = args[1]
            body = args[2]
        else:
            params = []
            body = args[1]
        # Convert parameter tokens to string names
        params = [t.value for t in params] if params else []
        return FunctionDecl(name, params, body)

    def param_list(self, args):
        return args  # list of Token identifiers

    def arg_list(self, args):
        return args  # list of Expressions

    def funapp(self, args):
        name = args[0].value
        if len(args) > 1:
            arg_list = args[1]
        else:
            arg_list = []
        return FunctionCall(name, arg_list)

    # Expression transformers
    def let_expr(self, args):
        name = args[0].value
        return LetExpr(name, args[1], args[2])

    def graph_bin(self, args):
        left = args[0]
        op = args[1].value   # the operator token (union, intersect, etc)
        right = args[2]
        return BinaryOp(op, left, right)

    def or_op(self, args):
        return BinaryOp('or', args[0], args[1])

    def and_op(self, args):
        return BinaryOp('and', args[0], args[1])

    def eq(self, args):
        return BinaryOp('==', args[0], args[1])

    def neq(self, args):
        return BinaryOp('!=', args[0], args[1])

    def lt(self, args):
        return BinaryOp('<', args[0], args[1])

    def gt(self, args):
        return BinaryOp('>', args[0], args[1])

    def le(self, args):
        return BinaryOp('<=', args[0], args[1])

    def ge(self, args):
        return BinaryOp('>=', args[0], args[1])

    def add(self, args):
        return BinaryOp('+', args[0], args[1])

    def sub(self, args):
        return BinaryOp('-', args[0], args[1])

    def mul(self, args):
        return BinaryOp('*', args[0], args[1])

    def div(self, args):
        return BinaryOp('/', args[0], args[1])

    def number(self, args):
        return Number(int(args[0].value))

    def true(self, args):
        return Bool(True)

    def false(self, args):
        return Bool(False)

    def var(self, args):
        return Var(args[0].value)

    def graphlit(self, args):
        # args[0].value includes quotes, strip them
        return GraphLiteral(args[0].value.strip("'"))

    def nodes(self, args):
        text = args[0].value
        inner = text.strip("{}")
        names = [s.strip() for s in inner.split(",") if s.strip()]
        return NodeSet(names)

# =====================================================================
# 4. State and Environment Setup
# =====================================================================
# We use an explicit store for state (heap) and an environment mapping names to locations.
class State:
    def __init__(self):
        self.store: dict[int, any] = {}  # location -> value
        self.next_loc: int = 0          # next free location index

def env_lookup(env: List[dict], name: str) -> int:
    """Find the location of 'name' by searching env frames from top."""
    for frame in reversed(env):
        if name in frame:
            return frame[name]
    raise ValueError(f"Undefined identifier: {name}")

# =====================================================================
# 5. Built-in Graph Operations and Utilities
# =====================================================================
# These functions perform graph manipulations as described.

def load_graph_from_file(filename: str) -> set:
    """Load a graph from a file. Each line 'u,v' is an undirected edge."""
    graph = set()
    fname = filename.strip("'")  # remove quotes if present
    try:
        with open(fname, "r") as f:
            for line in f:
                nodes = line.strip().split(",")
                if len(nodes) == 2:
                    graph.add(frozenset([nodes[0], nodes[1]]))
    except FileNotFoundError:
        print(f"File {fname} not found.")
    return graph

def union_graphs(graph1: set, graph2: set) -> set:
    return graph1.union(graph2)

def intersect_graphs(graph1: set, graph2: set) -> set:
    return graph1.intersection(graph2)

def difference_graphs(graph1: set, graph2: set) -> set:
    return graph1.difference(graph2)

def path_graphs(graph1: set, graph2: set) -> set:
    """Find any path (set of edges) connecting the two nodes in graph1 within graph2."""
    if len(graph1) != 1:
        raise ValueError("path operator requires a single-edge graph1")
    edge = next(iter(graph1))
    if len(edge) != 2:
        raise ValueError("Edge in graph1 must have two nodes")
    u, v = tuple(edge)
    # Build adjacency list for graph2
    adj: dict[str, Set[str]] = {}
    for e in graph2:
        if len(e) != 2:
            raise ValueError("graph2 edges must have two nodes")
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    # BFS from u to find v
    queue = deque([u])
    parent: dict[str, str | None] = {u: None}
    while queue:
        node = queue.popleft()
        if node == v:
            break
        for nbr in adj.get(node, []):
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    if v not in parent:
        return set()  # no path
    # Reconstruct path as edges
    path_edges = set()
    curr = v
    while parent[curr] is not None:
        prev = parent[curr]
        path_edges.add(frozenset([prev, curr]))
        curr = prev
    return path_edges

def shortest_path(graph1: set, graph2: set) -> set:
    """Find the shortest path (fewest edges) between the two nodes in graph1 within graph2."""
    if len(graph1) != 1:
        raise ValueError("shortest operator requires a single-edge graph1")
    edge = next(iter(graph1))
    if len(edge) != 2:
        raise ValueError("Edge in graph1 must have two nodes")
    s, t = tuple(edge)
    if s == t:
        return set()
    # Build adjacency list
    adj: dict[str, Set[str]] = {}
    for e in graph2:
        if len(e) != 2:
            raise ValueError("graph2 edges must have two nodes")
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    # BFS to find shortest path
    queue = deque([s])
    parent: dict[str, str | None] = {s: None}
    while queue:
        node = queue.popleft()
        if node == t:
            break
        for nbr in adj.get(node, []):
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    if t not in parent:
        return set()
    # Reconstruct path
    path_edges = set()
    curr = t
    while parent[curr] is not None:
        prev = parent[curr]
        path_edges.add(frozenset([prev, curr]))
        curr = prev
    return path_edges

# Visualization using networkx
def visualize_graph(graph: set):
    """Draw a graph (set of edges) using networkx and matplotlib."""
    G = nx.Graph()
    for edge in graph:
        u, v = tuple(edge)
        G.add_edge(u, v)
    nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10)
    plt.show()

# =====================================================================
# 6. Evaluation Semantics
# =====================================================================
def eval_expr(expr, env: List[dict], state: State):
    """Evaluate an expression AST node, returning its value. Does not mutate env (except via state for let/func)."""
    # Literal values
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, Bool):
        return expr.value
    # Graph literal: load from file
    if isinstance(expr, GraphLiteral):
        return load_graph_from_file(expr.filename)
    if isinstance(expr, NodeSet):
        # Interpret node set as empty graph with those isolated nodes
        return set()
    # Variable reference
    if isinstance(expr, Var):
        loc = env_lookup(env, expr.name)
        return state.store[loc]
    # Let-expression (local binding)
    if isinstance(expr, LetExpr):
        val = eval_expr(expr.expr, env, state)
        # Allocate new location for the bound variable
        loc = state.next_loc
        state.next_loc += 1
        state.store[loc] = val
        # Push a new frame for the variable
        env.append({expr.name: loc})
        result = eval_expr(expr.body, env, state)
        # Pop the frame and revert state
        env.pop()
        state.next_loc = loc
        return result
    # Binary operations
    if isinstance(expr, BinaryOp):
        left_val = eval_expr(expr.left, env, state)
        right_val = eval_expr(expr.right, env, state)
        op = expr.op
        # Graph operations
        if op == 'union':
            return union_graphs(left_val, right_val)
        if op == 'intersect':
            return intersect_graphs(left_val, right_val)
        if op == 'difference':
            return difference_graphs(left_val, right_val)
        if op == 'path':
            return path_graphs(left_val, right_val)
        if op == 'shortest':
            return shortest_path(left_val, right_val)
        # Arithmetic operations
        if op == '+':
            return left_val + right_val
        if op == '-':
            return left_val - right_val
        if op == '*':
            return left_val * right_val
        if op == '/':
            return left_val // right_val  # integer division
        # Relational
        if op == '==':
            return left_val == right_val
        if op == '!=':
            return left_val != right_val
        if op == '<':
            return left_val < right_val
        if op == '>':
            return left_val > right_val
        if op == '<=':
            return left_val <= right_val
        if op == '>=':
            return left_val >= right_val
        # Logical
        if op == 'and':
            return bool(left_val) and bool(right_val)
        if op == 'or':
            return bool(left_val) or bool(right_val)
        raise ValueError(f"Unknown operator: {op}")
    # Function call
    if isinstance(expr, FunctionCall):
        loc = env_lookup(env, expr.name)
        closure = state.store[loc]
        if not isinstance(closure, Closure):
            raise ValueError(f"{expr.name} is not a function")
        fn_decl = closure.function
        # Prepare new environment frames (static scoping)
        new_env = [frame.copy() for frame in closure.env]
        new_env.append({})  # frame for function parameters
        # Evaluate arguments
        arg_vals = [eval_expr(arg, env, state) for arg in expr.args]
        saved_next_loc = state.next_loc
        # Bind parameters
        for param, arg_val in zip(fn_decl.params, arg_vals):
            loc_new = state.next_loc
            state.next_loc += 1
            state.store[loc_new] = arg_val
            new_env[-1][param] = loc_new
        # Evaluate function body
        result = eval_expr(fn_decl.body, new_env, state)
        # Restore state (pop parameter frame)
        state.next_loc = saved_next_loc
        return result
    raise ValueError(f"Unknown expression type: {expr}")

def exec_command(cmd, env: List[dict], state: State):
    """Execute a single command AST node, returning possibly updated env and state."""
    # Variable declaration
    if isinstance(cmd, VarDecl):
        val = eval_expr(cmd.expr, env, state)
        loc = state.next_loc
        state.next_loc += 1
        state.store[loc] = val
        env[-1][cmd.name] = loc
        return env, state
    # Assignment
    if isinstance(cmd, Assign):
        val = eval_expr(cmd.expr, env, state)
        loc = env_lookup(env, cmd.name)
        state.store[loc] = val
        return env, state
    # Print command
    if isinstance(cmd, Print):
        val = eval_expr(cmd.expr, env, state)
        # Se è un grafo, disegna invece di stampare
        if isinstance(val, set) and all(isinstance(e, frozenset) for e in val):
            visualize_graph(val)
        else:
            print(val)
        return env, state
    # If-then-else
    if isinstance(cmd, If):
        cond_val = eval_expr(cmd.cond, env, state)
        if not isinstance(cond_val, bool):
            raise ValueError("If condition must be boolean")
        saved_loc = state.next_loc
        # Execute then or else branch with a fresh environment frame
        if cond_val:
            env.append({})
            env, state = exec_commands(cmd.then_branch, env, state)
        else:
            env.append({})
            env, state = exec_commands(cmd.else_branch, env, state)
        env.pop()
        state.next_loc = saved_loc
        return env, state
    # While loop
    if isinstance(cmd, While):
        while True:
            cond_val = eval_expr(cmd.cond, env, state)
            if not isinstance(cond_val, bool):
                raise ValueError("While condition must be boolean")
            if not cond_val:
                break
            saved_loc = state.next_loc
            env.append({})
            env, state = exec_commands(cmd.body, env, state)
            env.pop()
            state.next_loc = saved_loc
        return env, state
    # Function declaration
    if isinstance(cmd, FunctionDecl):
        # Capture the current environment frames
        captured_env = [frame.copy() for frame in env]
        closure = Closure(cmd, captured_env)
        loc = state.next_loc
        state.next_loc += 1
        state.store[loc] = closure
        env[-1][cmd.name] = loc
        return env, state
    raise ValueError(f"Unknown command type: {cmd}")

def exec_commands(cmds: List, env: List[dict], state: State):
    """Execute a list of commands sequentially."""
    for cmd in cmds:
        env, state = exec_command(cmd, env, state)
    return env, state

# =====================================================================
# 7. REPL / Test Harness (Example Programs)
# =====================================================================
if __name__ == "__main__":
    # Prepare initial environment and state
    # We start with a single global frame
    initial_env = [{}]
    state = State()

    # Example 1: Graph union and print
    print("EXAMPLE 1")
    example1 = "var g1 = 'graph1.txt'; var g2 = 'graph2.txt'; var g_union = g1 union g2; print g_union"
    ast1 = ASTTransformer().transform(parser.parse(example1))
    env1, state1 = exec_commands(ast1.commands, [dict(initial_env[0])], State())
    # Assume print outputs and also visualize the result

    # Example 2: Conditional selecting a graph
    print("EXAMPLE 2")
    example2 = "var x = 2; var g = 'graph1.txt'; if x > 1 then g <- 'graph1.txt' else g <- 'graph2.txt' endif; print g"
    ast2 = ASTTransformer().transform(parser.parse(example2))
    env2, state2 = exec_commands(ast2.commands, [dict(initial_env[0])], State())
    # The last expression 'g' is not printed by 'print', so we fetch it manually
    g_loc = env2[0]['g']

    # Example 3: While
    print("EXAMPLE 3")
    example3 = """
    var g_work   = 'graph1.txt';
    var endpoints = 'pair.txt';
    var result = g_work difference g_work;
    var k = 3;
    while k > 0 do
        var sp = endpoints shortest g_work;
        result <- result union sp;
        g_work <- g_work difference sp;
        k <- k - 1
    done;
    print result
    """
    ast3 = ASTTransformer().transform(parser.parse(example3))
    exec_commands(ast3.commands, [dict(initial_env[0])], State())

    # Example 4: Function
    print("EXAMPLE 4")
    example4 = """
    function overlay(a, b) = a union b;
    
    var g1 = 'graph1.txt';
    var g2 = 'graph2.txt';
    var merged = overlay(g1, g2);
    print merged
    """
    ast4 = ASTTransformer().transform(parser.parse(example4))
    exec_commands(ast4.commands, [dict(initial_env[0])], State())

    # Example 5: Function with graph operation
    print("EXAMPLE 5")
    example5 = "function cap(g,h) = g intersect h; var g1 = 'graph1.txt'; var g2 = 'graph4.txt'; var result = cap(g1, g2); print result"
    ast5 = ASTTransformer().transform(parser.parse(example5))
    env5, state5 = exec_commands(ast5.commands, [dict(initial_env[0])], State())