"""Deterministic audit and evaluation of LLM-written acquisition expressions.

The deliberately small language supports a single vectorized return expression.
It cannot access files, import packages, mutate features or alter FLOWR.
"""
import ast
import hashlib
import numpy as np


FUNCTIONS={'abs','tanh','clip','sqrt','minimum','maximum','exp'}
NODES=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,
       ast.Load,ast.Store,ast.Name,ast.Constant,ast.Subscript,ast.BinOp,ast.UnaryOp,
       ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.USub,ast.UAdd,
       ast.Call,ast.Attribute,ast.keyword)


def compile_rule(source,fields):
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('One score function required')
    fn=tree.body[0]
    if fn.name!='score' or fn.decorator_list or fn.returns or fn.type_comment:raise ValueError('Unsupported function declaration')
    if len(fn.args.args)!=1 or fn.args.args[0].arg!='features' or fn.args.args[0].annotation or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs:
        raise ValueError('Required signature: score(features)')
    if not 1<=len(fn.body)<=9 or not isinstance(fn.body[-1],ast.Return):raise ValueError('Use at most eight vector assignments and one return')
    locals=set()
    for statement in fn.body[:-1]:
        if not isinstance(statement,ast.Assign) or len(statement.targets)!=1 or not isinstance(statement.targets[0],ast.Name):raise ValueError('Only local vector assignments are allowed')
        name=statement.targets[0].id
        if name in {'np','features'} or name.startswith('_') or name in locals:raise ValueError('Reserved or repeated local name')
        if any(isinstance(n,ast.Name) and n.id not in {'np','features'}|locals for n in ast.walk(statement.value)):raise ValueError('Local used before definition')
        locals.add(name)
    used=set()
    for node in ast.walk(tree):
        if not isinstance(node,NODES):raise ValueError('Unsupported rule syntax: '+type(node).__name__)
        if isinstance(node,ast.Name) and node.id not in {'features','np'}|locals:raise ValueError('Unregistered name')
        if isinstance(node,ast.Constant) and not isinstance(node.value,(int,float,str)):raise ValueError('Unsupported constant')
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value not in fields:raise ValueError('Unregistered feature')
        if isinstance(node,ast.Subscript):
            if not isinstance(node.value,ast.Name) or node.value.id!='features' or not isinstance(node.slice,ast.Constant) or node.slice.value not in fields:
                raise ValueError('Only registered feature lookups are allowed')
            used.add(node.slice.value)
        if isinstance(node,ast.Attribute):
            if not isinstance(node.value,ast.Name) or node.value.id!='np' or node.attr not in FUNCTIONS:raise ValueError('Unregistered numerical function')
        if isinstance(node,ast.Call) and not isinstance(node.func,ast.Attribute):raise ValueError('Only whitelisted NumPy calls are allowed')
        if isinstance(node,ast.Call) and node.keywords:raise ValueError('No keyword outputs or mutation')
    if not used:raise ValueError('A rule must consume evidence features')
    scope={'__builtins__':{},'np':np}
    exec(compile(tree,'audited_acquisition_rule','exec'),scope)
    return scope['score'],sorted(used)


def evaluate_rule(source,features):
    function,used=compile_rule(source,features)
    with np.errstate(all='ignore'):result=np.asarray(function(features),dtype=float)
    n=len(next(iter(features.values())))
    if result.shape!=(n,) or not np.isfinite(result).all():raise ValueError('Rule must return one finite score per candidate')
    sensitivity={}
    for field in used:
        changed=dict(features);changed[field]=features[field]+.1
        with np.errstate(all='ignore'):other=np.asarray(function(changed),dtype=float)
        if other.shape!=(n,) or not np.isfinite(other).all():raise ValueError('Rule fails feature perturbation audit')
        sensitivity[field]=float(np.max(np.abs(other-result)))
    if max(sensitivity.values())<=1e-10:raise ValueError('Numerically inactive rule')
    return result,{'sha256':hashlib.sha256(source.encode()).hexdigest(),'used_features':used,
                   'perturbation_response':sensitivity,'score_range':[float(result.min()),float(result.max())]}
