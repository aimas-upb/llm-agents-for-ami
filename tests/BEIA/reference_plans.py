"""The hand-written ground-truth plan of each BEIA case, as the planner would emit it.

They are the solutions the cases' `ground_truth` facts describe, written in the
dependency compiler's IR (`interaction_solver/utils/dependency_compiler.py`):
the tree waits on blackboard keys, and the plan entry's runtime `reads` keep
those keys fresh from HASP. They serve two purposes:

  * a check of the scorer: each must validate, compile, and score `static_pass`
    against its own case (`tests/unit/test_beia_ground_truth.py`);
  * a reference for a person reading a failed row: what a right plan looks like.

For example A4, "whenever the garage AC turns on, close the garage window":

    repeat( sequence[ wait_for climate != off,
                      ignore_failure( close the window ),
                      wait_for climate == off ] )          <- re-arms the rule
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

HASP = "http://localhost:8080/workspaces"


# -- small IR builders -------------------------------------------------------------------

def prop(ref: str) -> str:
    """The HASP property URL of `workspace/artifact/name`."""
    ws, art, name = ref.split("/")
    return f"{HASP}/{ws}/artifacts/{art}/properties/{name}"


def act_url(ref: str) -> str:
    """The HASP action URL of `workspace/artifact/name`."""
    ws, art, name = ref.split("/")
    return f"{HASP}/{ws}/artifacts/{art}/actions/{name}"


def key(ref: str) -> str:
    """The blackboard key a property's reading is kept under (as the compiler names it)."""
    _, art, name = ref.split("/")
    return f"/artifacts/{art}/{name}"


def check(ref: str, op: str, value: Any) -> Dict[str, Any]:
    """One comparison of a property's reading, as a wait_for/check entry."""
    return {"key": key(ref), "op": op, "value": value}


def wait_for(name: str, checks: List[Dict[str, Any]], combine: str = "and_") -> Dict[str, Any]:
    """RUNNING until the checks hold (joined by `combine` when several)."""
    node = {"type": "wait_for", "name": name, "checks": checks}
    if len(checks) > 1:
        node["combine"] = combine
    return node


def action(name: str, ref: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Invoke a HASP action with these parameters."""
    return {"type": "action", "name": name, "action_url": act_url(ref),
            "parameters": params or {}}


def alert(name: str, text: str) -> Dict[str, Any]:
    """Raise an alert to the user (P2's node)."""
    return {"type": "alert", "name": name, "text": text}


def seq(name: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The children in order, remembering progress across ticks."""
    return {"type": "sequence", "name": name, "memory": True, "children": children}


def one(kind: str, name: str, child: Dict[str, Any]) -> Dict[str, Any]:
    """A decorator node around one child."""
    return {"type": kind, "name": name, "children": [child]}


def whenever(name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A standing rule: the steps, again and again."""
    return one("repeat", f"{name}: whenever", seq(f"{name}: once", steps))


def held(name: str, holds: Dict[str, Any], breaks: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    """RUNNING until `holds` has stayed true `seconds` without a break (P3's block):
    inverter(repeat(inverter(sequence[wait holds, parallel[timer, inverter(wait breaks)]])))."""
    attempt = seq(f"{name}: attempt", [
        wait_for(f"{name}: holds", [holds]),
        {"type": "parallel", "name": f"{name}: race", "policy": "success_on_one", "children": [
            {"type": "timer", "name": f"{name}: span", "seconds": float(seconds)},
            one("inverter", f"{name}: broke", wait_for(f"{name}: breaks", [breaks]))]}])
    return one("inverter", f"{name}: held",
               one("repeat", f"{name}: retry",
                   one("inverter", f"{name}: until it holds", attempt)))


def plan(tree: Dict[str, Any], reads: List[str], clock: bool = False) -> Dict[str, Any]:
    """The UA's {"plans": [entry]} for one dependency entry."""
    return {"plans": [{"tree": tree, "runtime": {
        "reads": [{"key": key(r), "url": prop(r), "field": None} for r in reads],
        "derived": [], "clock": clock}}]}


# -- the six cases -----------------------------------------------------------------------

PM25 = "garaj/blue_air_garage/pm25"
FAN = "garaj/blue_air_garage"
GRID = "solaredge/solaredge_i4_m1/ac_power"
LG = "garaj/lg_garaj_hvac/climate"
NETATMO_RH = "garaj/weather_station_70_ee_50_2e_da_88/humidity"
NETATMO_T = "garaj/weather_station_70_ee_50_2e_da_88/temperature"


def a1() -> Dict[str, Any]:
    """PM2.5 above 25 for 5 min -> full speed; below 12 for 10 min -> Auto."""
    return plan(whenever("A1", [
        held("rise", check(PM25, "gt", 25), check(PM25, "le", 25), 300),
        action("full speed", f"{FAN}/fan_set_percentage", {"percentage": 100}),
        held("fall", check(PM25, "lt", 12), check(PM25, "ge", 12), 600),
        action("back to auto", f"{FAN}/fan_set_preset_mode", {"preset_mode": "auto"}),
    ]), [PM25], clock=True)


def a2() -> Dict[str, Any]:
    """Exporting and PM2.5 above 25 -> full speed; re-armed when either stops."""
    return plan(whenever("A2", [
        wait_for("surplus and dusty", [check(GRID, "gt", 0), check(PM25, "gt", 25)]),
        one("ignore_failure", "a failed run does not stop later ones",
            action("full speed", f"{FAN}/fan_set_percentage", {"percentage": 100})),
        wait_for("either stopped", [check(GRID, "le", 0), check(PM25, "le", 25)], "or_"),
    ]), [GRID, PM25])


def a3() -> Dict[str, Any]:
    """Import or export above 2 kW -> alert; re-armed back in band."""
    return plan(whenever("A3", [
        wait_for("out of band", [check(GRID, "lt", -2000), check(GRID, "gt", 2000)], "or_"),
        alert("G1: alert", "SolarEdge grid import/export above 2 kW"),
        wait_for("back in band", [check(GRID, "ge", -2000), check(GRID, "le", 2000)]),
    ]), [GRID])


def a4() -> Dict[str, Any]:
    """The garage AC turns on -> close the garage window."""
    return plan(whenever("A4", [
        wait_for("AC on", [check(LG, "ne", "off")]),
        one("ignore_failure", "a failed run does not stop later ones",
            action("close the window", "garaj/actuator_geam_2/cover_close_cover")),
        wait_for("AC off", [check(LG, "eq", "off")]),
    ]), [LG])


def a5() -> Dict[str, Any]:
    """Above 65 % -> dehumidifier on; below 60 % -> off."""
    return plan(whenever("A5", [
        wait_for("too damp", [check(NETATMO_RH, "gt", 65)]),
        action("dehumidify", f"{FAN}/dehumidifier_turn_on"),
        wait_for("dry enough", [check(NETATMO_RH, "lt", 60)]),
        action("stop", f"{FAN}/dehumidifier_turn_off"),
    ]), [NETATMO_RH])


def a5_native() -> Dict[str, Any]:
    """The other right answer: the dehumidifier holds the band itself."""
    return plan(seq("A5 native", [
        action("target 60 %", f"{FAN}/dehumidifier_set_humidity", {"humidity": 60}),
        action("dehumidify", f"{FAN}/dehumidifier_turn_on"),
    ]), [])


def a6() -> Dict[str, Any]:
    """Above 26 °C -> the AC cooling at 21 °C; re-armed once below."""
    return plan(whenever("A6", [
        wait_for("too warm", [check(NETATMO_T, "gt", 26)]),
        one("ignore_failure", "a failed run does not stop later ones", seq("cool to 21", [
            action("cool", "garaj/lg_garaj_hvac/climate_set_hvac_mode", {"hvac_mode": "cool"}),
            action("21 degrees", "garaj/lg_garaj_hvac/climate_set_temperature",
                   {"temperature": 21})])),
        wait_for("not too warm", [check(NETATMO_T, "le", 26)]),
    ]), [NETATMO_T])


REFERENCE_PLANS = {
    "A1": [a1], "A2": [a2], "A3": [a3], "A4": [a4], "A5": [a5, a5_native], "A6": [a6],
}
