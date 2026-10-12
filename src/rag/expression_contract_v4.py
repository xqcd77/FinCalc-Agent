"""V4 restricted Decimal expression contract; independent of frozen V2/V3."""
import ast
from decimal import Decimal, localcontext, ROUND_HALF_UP
from financial_semantics_v4 import V4Error
ALLOWED_CONSTANTS={Decimal(x) for x in (*range(13),360,365)}
MAX_EXPRESSION_CHARS=1200
MAX_EXPRESSION_NODES=512

def validate_expression(expression, dimensions):
    """Same whitelist/signatures/dimensions as execution, without dummy numerical math."""
    if not isinstance(expression,str) or not expression.strip() or len(expression)>MAX_EXPRESSION_CHARS:
        raise V4Error('expression','表达式缺失或过长')
    try: tree=ast.parse(expression,mode='eval')
    except SyntaxError as exc: raise V4Error('expression','表达式语法错误') from exc
    if sum(1 for _ in ast.walk(tree))>MAX_EXPRESSION_NODES:raise V4Error('expression','表达式节点过多')
    def walk(node,depth=0):
        if depth>24:raise V4Error('expression','表达式层级过深')
        if isinstance(node,ast.Expression):return walk(node.body,depth+1)
        if isinstance(node,ast.Name) and node.id in dimensions:return dimensions[node.id]
        if isinstance(node,ast.Constant) and type(node.value) in (int,float) and Decimal(str(node.value)) in ALLOWED_CONSTANTS:return (0,0)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return walk(node.operand,depth+1)
        if isinstance(node,ast.BinOp):
            a,b=walk(node.left,depth+1),walk(node.right,depth+1)
            if isinstance(node.op,(ast.Add,ast.Sub)):
                if a!=b:raise V4Error('expression','加减的财务维度不一致')
                return a
            if isinstance(node.op,ast.Mult):return (a[0]+b[0],a[1]+b[1])
            if isinstance(node.op,ast.Div):return (a[0]-b[0],a[1]-b[1])
            if isinstance(node.op,ast.Pow) and a==b==(0,0):return (0,0)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            fn=node.func.id;ds=[walk(x,depth+1) for x in node.args]
            if fn=='round_cent':
                if len(ds)!=1 or ds[0]!=(1,-1):raise V4Error('expression','round_cent只接受一个元/股参数；比率不要调用它，最终统一四舍五入')
                return ds[0]
            if fn in {'min','max'} and ds and len(set(ds))==1:return ds[0]
            if fn in {'nth_largest','nth_smallest'} and len(ds)>=2 and ds[0]==(0,0) and len(set(ds[1:]))==1:
                rank=node.args[0]
                if not isinstance(rank,ast.Constant) or type(rank.value)!=int or not 1<=rank.value<len(ds):raise V4Error('expression','排序位置必须为合法固定整数')
                return ds[1]
        raise V4Error('expression','不允许的表达式节点、常量或函数参数')
    return walk(tree)

def calculate_typed(expression: str, values: dict[str, Decimal],
                    dimensions: dict[str, tuple[int, int]]) -> tuple[Decimal, tuple[int, int]]:
    if not isinstance(expression, str) or len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("Expression missing or too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree))>MAX_EXPRESSION_NODES:raise ValueError('Expression has too many nodes')

    def walk(node: ast.AST, depth: int = 0) -> tuple[Decimal, tuple[int, int]]:
        if depth > 24:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id], dimensions[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = Decimal(str(node.value))
            if literal not in ALLOWED_CONSTANTS:
                raise ValueError("Unattributed numeric literal; use a sourced operand")
            return literal, (0, 0)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value, dim = walk(node.operand, depth + 1)
            return (value if isinstance(node.op, ast.UAdd) else -value), dim
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            function = node.func.id
            if function in {"min", "max", "nth_largest", "nth_smallest"}:
                args = [walk(arg, depth + 1) for arg in node.args]
                rank = 1
                if function.startswith("nth_"):
                    if len(args) < 2 or args[0][1] != (0, 0) or args[0][0] != args[0][0].to_integral_value():
                        raise ValueError("Order statistic needs an integer rank and values")
                    rank = int(args.pop(0)[0])
                if not args or not 1 <= rank <= len(args) or len({dim for _, dim in args}) != 1:
                    raise ValueError("Order statistic needs comparable values and valid rank")
                descending = function in {"max", "nth_largest"}
                ordered = sorted((value for value, _ in args), reverse=descending)
                return ordered[rank - 1], args[0][1]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round_cent" and not node.keywords and len(node.args) == 1:
            value, dim = walk(node.args[0], depth + 1)
            if dim != (1, -1):
                raise ValueError("round_cent requires money per share")
            return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), dim
        if isinstance(node, ast.BinOp):
            left, ld = walk(node.left, depth + 1)
            right, rd = walk(node.right, depth + 1)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if ld != rd and not ((left == 0 and ld == (0, 0)) or (right == 0 and rd == (0, 0))):
                    raise ValueError("Expression adds incompatible financial units")
                dim = rd if left == 0 and ld == (0, 0) else ld
                return (left + right if isinstance(node.op, ast.Add) else left - right), dim
            if isinstance(node.op, ast.Mult):
                return left * right, (ld[0] + rd[0], ld[1] + rd[1])
            if isinstance(node.op, ast.Div):
                return left / right, (ld[0] - rd[0], ld[1] - rd[1])
            if isinstance(node.op, ast.Pow):
                if rd != (0, 0) or abs(right) > 12 or (ld != (0, 0) and right != 1):
                    raise ValueError("Invalid power or dimensioned exponentiation")
                return left ** right, ld
        raise ValueError("Unsupported expression syntax")

    with localcontext() as ctx:
        ctx.prec = 40
        value, dim = walk(tree)
        if not value.is_finite() or abs(value) > Decimal("1e30"):
            raise ValueError("Result out of bounds")
        return +value, dim


