"""
Agents — multi-agent layer for the street-environment safety & comfort
evaluation tool.

    Orchestrator         — workflow coordination
    Physical_Auditor     — physical facility data (CCTV / emergency bell / security light)
    Perceptual_Observer  — perceptual attribute data (Street View + DeepLabV3+)
    Evaluator            — I_phy / I_per z-score + matrix quadrant classification

Lazy import: agents are loaded on first access to avoid pulling heavy deps
(e.g. torch) when only one agent is needed.
"""

__all__ = [
    "Orchestrator",
    "Physical_Auditor", "Perceptual_Observer", "Evaluator",
]

_MODULE_MAP = {
    "Physical_Auditor":    "agents.physical_auditor",
    "Perceptual_Observer": "agents.perceptual_observer",
    "Evaluator":           "agents.evaluator",
    "Orchestrator":        "agents.orchestrator",
}

def __getattr__(name):
    if name in _MODULE_MAP:
        import importlib
        mod = importlib.import_module(_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
