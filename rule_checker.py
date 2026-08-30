"""
src/rule_checker.py
-------------------
Deterministic rule checker for Cisco IOS show-command output.

Each rule is a pure function:
  check_<name>(text: str) -> list[Finding]

A Finding is a small dataclass with fields:
  rule_id, severity, message, matched_text, recommendation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    rule_id: str
    severity: str          # "Critical" | "High" | "Medium" | "Low" | "Info"
    message: str
    matched_text: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "matched_text": self.matched_text,
            "recommendation": self.recommendation,
        }


# ── Individual rule functions ──────────────────────────────────────────────────

def check_admin_down(text: str) -> list[Finding]:
    """
    Detects interfaces that are administratively shut down.
    Matches: 'FastEthernet0/0 is administratively down'
    """
    findings: list[Finding] = []
    pattern = re.compile(
        r"(\S+)\s+is\s+administratively\s+down",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        iface = match.group(1)
        findings.append(Finding(
            rule_id="RC-001",
            severity="High",
            message=f"Interface {iface} is administratively down (shutdown applied).",
            matched_text=match.group(0),
            recommendation=f"Run: interface {iface} → no shutdown",
        ))
    return findings


def check_line_protocol_down(text: str) -> list[Finding]:
    """
    Detects interfaces where line protocol is down but interface is not admin-down.
    Matches: 'GigabitEthernet0/1 is up, line protocol is down'
    """
    findings: list[Finding] = []
    pattern = re.compile(
        r"(\S+)\s+is\s+up,\s+line\s+protocol\s+is\s+down",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        iface = match.group(1)
        findings.append(Finding(
            rule_id="RC-002",
            severity="High",
            message=f"Interface {iface} is up but line protocol is down — likely a Layer 1/2 issue.",
            matched_text=match.group(0),
            recommendation="Check physical cable, check clock rate on serial (DCE side), check encapsulation.",
        ))
    return findings


def check_no_default_gateway(text: str) -> list[Finding]:
    """
    Detects when 'Gateway of last resort is not set' appears in routing table output.
    """
    findings: list[Finding] = []
    if re.search(r"Gateway of last resort is not set", text, re.IGNORECASE):
        findings.append(Finding(
            rule_id="RC-003",
            severity="Critical",
            message="No default gateway (gateway of last resort) is configured on this router.",
            matched_text="Gateway of last resort is not set",
            recommendation="Add: ip route 0.0.0.0 0.0.0.0 <next-hop-ip>",
        ))
    return findings


def check_missing_dhcp_default_router(text: str) -> list[Finding]:
    """
    Detects DHCP pool blocks missing a 'default-router' statement.
    Parses 'ip dhcp pool' config blocks.
    """
    findings: list[Finding] = []
    # Find all dhcp pool blocks
    pool_blocks = re.findall(
        r"ip dhcp pool\s+(\S+)(.*?)(?=\nip dhcp pool|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    for pool_name, pool_body in pool_blocks:
        if not re.search(r"default-router", pool_body, re.IGNORECASE):
            findings.append(Finding(
                rule_id="RC-004",
                severity="High",
                message=f"DHCP pool '{pool_name}' is missing a default-router statement.",
                matched_text=f"ip dhcp pool {pool_name}",
                recommendation=f"Under 'ip dhcp pool {pool_name}': add 'default-router <gateway-ip>'",
            ))
    return findings


def check_native_vlan_mismatch(text: str) -> list[Finding]:
    """
    Detects native VLAN mismatches in CDP or trunk output.
    Matches: CDP warning about native VLAN mismatch.
    """
    findings: list[Finding] = []
    pattern = re.compile(
        r"native\s+vlan\s+mismatch|%CDP-4-NATIVE_VLAN_MISMATCH",
        re.IGNORECASE,
    )
    if pattern.search(text):
        findings.append(Finding(
            rule_id="RC-005",
            severity="Medium",
            message="Native VLAN mismatch detected on a trunk link.",
            matched_text="Native VLAN mismatch",
            recommendation="Ensure both ends of the trunk share the same native VLAN: 'switchport trunk native vlan <id>'",
        ))
    return findings


def check_err_disabled(text: str) -> list[Finding]:
    """
    Detects ports in err-disabled state (e.g., from port security violation).
    """
    findings: list[Finding] = []
    pattern = re.compile(
        r"(\S+)\s+.*?err-disabled",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        findings.append(Finding(
            rule_id="RC-006",
            severity="High",
            message=f"Port {match.group(1)} is in err-disabled state.",
            matched_text=match.group(0).strip(),
            recommendation="Identify violation cause, then: shutdown → no shutdown on the interface to re-enable.",
        ))
    return findings


def check_acl_deny_all(text: str) -> list[Finding]:
    """
    Detects explicit 'deny ip any any' or 'deny any' at the top of an ACL
    that would block all traffic.
    """
    findings: list[Finding] = []
    pattern = re.compile(
        r"^\s*\d+\s+deny\s+ip\s+any\s+any",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(text):
        findings.append(Finding(
            rule_id="RC-007",
            severity="Critical",
            message="ACL contains an explicit 'deny ip any any' rule — all traffic may be blocked.",
            matched_text=match.group(0).strip(),
            recommendation="Review ACL order. Ensure permit statements precede catch-all deny rules. Use 'ip access-list resequence' if needed.",
        ))
    return findings


def check_ospf_no_neighbors(text: str) -> list[Finding]:
    """
    Detects OSPF neighbor table with no entries (FULL state missing).
    """
    findings: list[Finding] = []
    # If 'show ip ospf neighbor' output present but no FULL state shown
    has_ospf_header = re.search(r"Neighbor ID.*Dead Time.*Address.*Interface", text, re.IGNORECASE)
    has_full_state = re.search(r"\bFULL\b", text, re.IGNORECASE)

    if has_ospf_header and not has_full_state:
        findings.append(Finding(
            rule_id="RC-008",
            severity="High",
            message="OSPF neighbor table shows no neighbors in FULL state — adjacency is not forming.",
            matched_text="OSPF neighbor table: no FULL state",
            recommendation="Verify: matching area IDs, hello/dead timers, network statements, authentication settings, and MTU.",
        ))
    return findings


def check_vlan_not_created(text: str) -> list[Finding]:
    """
    Detects when a port is assigned to a VLAN that doesn't appear in 'show vlan brief'.
    Heuristic: access mode VLAN set to a VLAN not in the active VLAN list.
    """
    findings: list[Finding] = []
    # Extract VLANs that are 'active'
    active_vlans = set(re.findall(r"^(\d+)\s+\S+\s+active", text, re.IGNORECASE | re.MULTILINE))
    # Extract access mode VLANs from switchport output
    access_vlans = re.findall(r"Access Mode VLAN:\s*(\d+)", text, re.IGNORECASE)
    for vlan in access_vlans:
        if active_vlans and vlan not in active_vlans:
            findings.append(Finding(
                rule_id="RC-009",
                severity="Medium",
                message=f"Port is assigned to VLAN {vlan} but that VLAN does not appear as active in 'show vlan brief'.",
                matched_text=f"Access Mode VLAN: {vlan}",
                recommendation=f"Create the VLAN first: 'vlan {vlan}' then 'name <NAME>' in global config mode.",
            ))
    return findings


def check_isakmp_no_state(text: str) -> list[Finding]:
    """
    Detects IKEv1 Phase 1 failure: MM_NO_STATE in ISAKMP SA table.
    """
    findings: list[Finding] = []
    if re.search(r"MM_NO_STATE|QM_IDLE.*deleted", text, re.IGNORECASE):
        findings.append(Finding(
            rule_id="RC-010",
            severity="Critical",
            message="IKEv1 Phase 1 is failing (MM_NO_STATE). VPN tunnel will not establish.",
            matched_text="MM_NO_STATE",
            recommendation="Verify ISAKMP policy: encryption, hash, auth method, DH group, and lifetime must match on both peers. Also check pre-shared key.",
        ))
    return findings


def check_duplicate_ip(text: str) -> list[Finding]:
    """
    Detects Cisco duplicate IP address warning messages.
    """
    findings: list[Finding] = []
    pattern = re.compile(
        r"Duplicate address (\d+\.\d+\.\d+\.\d+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        findings.append(Finding(
            rule_id="RC-011",
            severity="Critical",
            message=f"Duplicate IP address detected: {match.group(1)}",
            matched_text=match.group(0),
            recommendation=f"Identify which device is wrongly assigned {match.group(1)} and change it to a unique address.",
        ))
    return findings


def check_voice_vlan_missing(text: str) -> list[Finding]:
    """
    Detects access port where Voice VLAN is set to 'none' despite being a
    likely VoIP-facing port (has 'Voice VLAN:' line).
    """
    findings: list[Finding] = []
    pattern = re.compile(r"Voice VLAN:\s*(none)", re.IGNORECASE)
    for match in pattern.finditer(text):
        findings.append(Finding(
            rule_id="RC-012",
            severity="Medium",
            message="Access port has Voice VLAN set to 'none' — VoIP phones will not tag to a voice VLAN.",
            matched_text=match.group(0),
            recommendation="Configure: 'switchport voice vlan <vlan-id>' on the access port.",
        ))
    return findings


# ── Registry and runner ────────────────────────────────────────────────────────

# All registered rule functions in priority order
RULES: list[Callable[[str], list[Finding]]] = [
    check_admin_down,
    check_line_protocol_down,
    check_no_default_gateway,
    check_missing_dhcp_default_router,
    check_native_vlan_mismatch,
    check_err_disabled,
    check_acl_deny_all,
    check_ospf_no_neighbors,
    check_vlan_not_created,
    check_isakmp_no_state,
    check_duplicate_ip,
    check_voice_vlan_missing,
]

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


def run_all_checks(show_output: str) -> list[Finding]:
    """
    Run every registered rule against the provided show-command output string.

    Returns a list of Finding objects sorted by severity (Critical first).
    """
    all_findings: list[Finding] = []
    for rule_fn in RULES:
        try:
            all_findings.extend(rule_fn(show_output))
        except Exception as exc:
            # Never let a buggy rule crash the whole checker
            all_findings.append(Finding(
                rule_id="RC-ERR",
                severity="Info",
                message=f"Rule '{rule_fn.__name__}' raised an error: {exc}",
            ))

    # De-duplicate by (rule_id, matched_text) to avoid repeated identical findings
    seen: set[tuple[str, str]] = set()
    deduped: list[Finding] = []
    for f in all_findings:
        key = (f.rule_id, f.matched_text)
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    deduped.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 99))
    return deduped


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    return [f.to_dict() for f in findings]
