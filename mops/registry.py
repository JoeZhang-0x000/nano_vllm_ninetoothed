from typing import Callable, Dict

# 全局注册表
_BACKEND_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "triton": {},
    "ninetoothed": {},
    "torch": {},
}

_CURRENT_BACKEND = None


def set_current_backend(backend: str):
    """设置当前后端"""
    global _CURRENT_BACKEND
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(f"Backend {backend} not supported")
    _CURRENT_BACKEND = backend


def register_op(backend: str):
    """算子注册装饰器"""
    def decorator(func: Callable) -> Callable:
        op_name = func.__name__
        if op_name in _BACKEND_REGISTRY[backend]:
            raise RuntimeError(f"Operator {op_name} already registered for {backend}")
        _BACKEND_REGISTRY[backend][op_name] = func
        return func
    return decorator


def get_op(op_name: str, backend: str=None) -> Callable:
    """获取当前后端的算子"""
    if backend is None:
        backend = _CURRENT_BACKEND
    op = _BACKEND_REGISTRY[backend].get(op_name)
    if op is None:
        print(f"Operator [{op_name}] not found in {backend} backend")
    return op