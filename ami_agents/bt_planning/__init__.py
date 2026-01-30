"""
BehaviorTree Planning Module

Ported from behaviortree-planning-for-ami-agents for integration into
the AAMAS 2026 demo multi-agent system.

Provides:
- nodes/: py_trees node implementations for HMAS affordances
- planning/: BT JSON IR generation via LLM
- execution/: JSON IR compilation to py_trees and tick-loop execution
- signifier_bridge: BT <-> Signifier conversion utilities
"""
