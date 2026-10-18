# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release piecewise-submodule inputs at Inductor's last use.

vLLM's stitching code calls each compiled piece as ``out = submod(a, b, c)`` and only
``del``s ``a, b, c`` after the call returns, so attention outputs, gates and the old
residual that a piece consumes in its first kernels stay allocated through its MLP.
The rewrite passes the inputs whose last use is that call in a list that only the
callee owns; AOTAutograd steals the list and the Inductor ``call(args)`` clears it and
``del``s each input after its last kernel. Kernels, order and values are unchanged.
"""

import ast
import inspect
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.piecewise_backend import PiecewiseBackend
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

BOX = "__vllm_boxed_args__"
CALL = "__vllm_call_boxed__"
SUBMODS = "__vllm_submods__"
_inner: dict[int, Any] = {}
_warned: set[str] = set()


def rewrite(code: str) -> tuple[str, int]:
    """Rewrite ``execution_fn`` so stolen inputs are only referenced by the box."""
    tree = ast.parse(code)
    rewritten = 0
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "execution_fn":
            continue
        body: list[ast.stmt] = []
        statements = function.body
        index = 0
        while index < len(statements):
            statement = statements[index]
            following = statements[index + 1] if index + 1 < len(statements) else None
            call = statement.value if isinstance(statement, ast.Assign) else None
            if (
                isinstance(statement, ast.Assign)
                and isinstance(call, ast.Call)
                and not call.keywords
                and isinstance(call.func, ast.Subscript)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == SUBMODS
                and isinstance(following, ast.Delete)
            ):
                deleted = [
                    target.id
                    for target in following.targets
                    if isinstance(target, ast.Name)
                ]
                passed = {arg.id for arg in call.args if isinstance(arg, ast.Name)}
                stolen = [name for name in deleted if name in passed]
                if stolen and len(deleted) == len(following.targets):
                    rest = [name for name in deleted if name not in passed]
                    body.append(
                        ast.Assign(
                            targets=[ast.Name(BOX, ast.Store())],
                            value=ast.List(elts=call.args, ctx=ast.Load()),
                        )
                    )
                    body.append(
                        ast.Delete(
                            targets=[ast.Name(name, ast.Del()) for name in stolen]
                        )
                    )
                    statement.value = ast.Call(
                        func=ast.Name(CALL, ast.Load()),
                        args=[call.func, ast.Name(BOX, ast.Load())],
                        keywords=[],
                    )
                    body.append(statement)
                    body.append(
                        ast.Delete(
                            targets=[ast.Name(name, ast.Del()) for name in [BOX, *rest]]
                        )
                    )
                    rewritten += 1
                    index += 2
                    continue
            body.append(statement)
            index += 1
        function.body = body
    return ast.unparse(ast.fix_missing_locations(tree)), rewritten


def _warn_once(reason: str) -> None:
    if reason not in _warned:
        _warned.add(reason)
        logger.warning("Boxed piecewise call fell back to tuple args: %s", reason)


def _boxed_inner(runnable: Any) -> Any | None:
    key = id(runnable)
    cached = _inner.get(key)
    if cached is not None and cached[0] is runnable:
        return cached[1]
    inner = None
    try:
        # Unpack only PiecewiseBackend.get_compiled_graph_wrapper's cached artifact;
        # fresh-compile runnables keep their own call path.
        if not getattr(runnable, "__qualname__", "").endswith(
            "get_compiled_graph_wrapper.<locals>.compiled_graph_wrapper"
        ):
            raise TypeError("not a PiecewiseBackend cached-artifact wrapper")
        artifact = inspect.getclosurevars(runnable).nonlocals["compiled_graph"]
        forward = artifact.inner_fn.compiled_fn
        candidate = forward.__wrapped__
        if type(candidate).__name__ == "SerializableCompiledFunction":
            inner = candidate
        else:
            _warn_once(f"unexpected inner {type(candidate).__qualname__}")
    except (AttributeError, KeyError, TypeError) as error:
        _warn_once(f"{type(runnable).__qualname__}: {error}")
    _inner[key] = (runnable, inner)
    return inner


def _tuple_call(function: Any, box: list[Any]) -> Any:
    args = tuple(box)
    box.clear()
    return function(*args)


def call_boxed(submod: Any, box: list[Any]) -> Any:
    backend = submod
    if isinstance(backend, CUDAGraphWrapper):
        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode != CUDAGraphMode.NONE and mode == backend.runtime_mode:
                return _tuple_call(submod, box)
        backend = backend.runnable
    if not isinstance(backend, PiecewiseBackend):
        _warn_once(f"submodule {type(backend).__qualname__}")
        return _tuple_call(backend, box)

    # Same range selection as PiecewiseBackend.__call__.
    if backend.sym_shape_indices:
        range_entry = backend._find_range_for_shape(box[backend.sym_shape_indices[0]])
        assert range_entry is not None
    else:
        compiled = [entry for entry in backend.range_entries.values() if entry.compiled]
        assert len(compiled) == 1
        range_entry = compiled[0]
    assert range_entry.compiled
    inner = _boxed_inner(range_entry.runnable)
    if inner is None:
        return _tuple_call(range_entry.runnable, box)
    logger.info_once("Boxed piecewise calls engaged")
    output = inner(box)
    # Same unpacking as PiecewiseBackend.get_compiled_graph_wrapper.
    if backend.returns_tuple or not isinstance(output, (tuple, list)):
        return output
    return output[0]
