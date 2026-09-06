"""
topology.py
-----------
Defines the simulated MPLS / SD-WAN network: which devices exist,
what role each plays, and how they are wired together.

Device roles (standard MPLS terminology):
    CE  - Customer Edge   : sits at a branch/site, hands traffic to the provider
    PE  - Provider Edge   : the provider router a CE connects to; does MPLS labelling
    P   - Provider core   : pure MPLS backbone router, label switching only

A real enterprise SD-WAN-over-MPLS deployment looks like:
    branch CE --- PE --- P --- P --- PE --- datacenter CE
with IPSec overlay tunnels riding on top of that underlay.

We model this as a graph so the predictive engine (later phase) can do
graph-based event correlation: "if this P router degrades, which sites lose service?"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class Role(str, Enum):
    CE = "CE"   # customer edge (a site)
    PE = "PE"   # provider edge
    P = "P"     # provider core


class SiteType(str, Enum):
    BRANCH = "branch"
    HUB = "hub"
    DATACENTER = "datacenter"


@dataclass
class Node:
    """A single network device."""
    name: str
    role: Role
    site_type: SiteType | None = None   # only CE nodes belong to a site
    # baseline health characteristics — used by the telemetry layer
    base_latency_ms: float = 5.0        # nominal one-hop latency
    capacity_mbps: float = 1000.0       # interface capacity

    def __str__(self) -> str:
        loc = f"/{self.site_type.value}" if self.site_type else ""
        return f"{self.name}({self.role.value}{loc})"


@dataclass
class Link:
    """A directed-or-bidirectional connection between two nodes."""
    a: str            # node name
    b: str            # node name
    capacity_mbps: float = 1000.0
    base_latency_ms: float = 3.0
    is_tunnel: bool = False   # True = IPSec overlay tunnel, False = physical underlay

    def key(self) -> Tuple[str, str]:
        # canonical key so (a,b) and (b,a) are treated as the same link
        return tuple(sorted((self.a, self.b)))  # type: ignore


@dataclass
class Topology:
    """The whole network: a collection of nodes and links."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    links: List[Link] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_link(self, link: Link) -> None:
        self.links.append(link)

    def neighbors(self, node_name: str) -> List[str]:
        """All nodes directly connected to the given node."""
        out = []
        for ln in self.links:
            if ln.a == node_name:
                out.append(ln.b)
            elif ln.b == node_name:
                out.append(ln.a)
        return out

    def sites(self) -> List[Node]:
        """All CE nodes (the actual customer sites)."""
        return [n for n in self.nodes.values() if n.role == Role.CE]

    def summary(self) -> str:
        n_ce = sum(1 for n in self.nodes.values() if n.role == Role.CE)
        n_pe = sum(1 for n in self.nodes.values() if n.role == Role.PE)
        n_p = sum(1 for n in self.nodes.values() if n.role == Role.P)
        n_tun = sum(1 for ln in self.links if ln.is_tunnel)
        return (
            f"Topology: {len(self.nodes)} nodes "
            f"({n_ce} CE, {n_pe} PE, {n_p} P), "
            f"{len(self.links)} links ({n_tun} tunnels)"
        )


def build_default_topology() -> Topology:
    """
    Builds a small but realistic multi-site topology:

        Branch-1 (CE) --- PE-1 ---\
                                   P-1 --- P-2 --- PE-3 --- Datacenter (CE)
        Branch-2 (CE) --- PE-2 ---/

    Plus IPSec overlay tunnels from each branch to the datacenter,
    which is how SD-WAN actually carries application traffic.
    """
    topo = Topology()

    # --- Customer sites (CE) ---
    topo.add_node(Node("branch1", Role.CE, SiteType.BRANCH, base_latency_ms=8))
    topo.add_node(Node("branch2", Role.CE, SiteType.BRANCH, base_latency_ms=9))
    topo.add_node(Node("hub", Role.CE, SiteType.HUB, base_latency_ms=6))
    topo.add_node(Node("datacenter", Role.CE, SiteType.DATACENTER,
                       base_latency_ms=4, capacity_mbps=10000))

    # --- Provider edge (PE) ---
    topo.add_node(Node("pe1", Role.PE, base_latency_ms=3))
    topo.add_node(Node("pe2", Role.PE, base_latency_ms=3))
    topo.add_node(Node("pe3", Role.PE, base_latency_ms=3, capacity_mbps=10000))

    # --- Provider core (P) ---
    topo.add_node(Node("p1", Role.P, base_latency_ms=2, capacity_mbps=10000))
    topo.add_node(Node("p2", Role.P, base_latency_ms=2, capacity_mbps=10000))

    # --- Underlay (physical) links ---
    topo.add_link(Link("branch1", "pe1", capacity_mbps=1000, base_latency_ms=4))
    topo.add_link(Link("branch2", "pe2", capacity_mbps=1000, base_latency_ms=4))
    topo.add_link(Link("hub", "pe1", capacity_mbps=2000, base_latency_ms=3))
    topo.add_link(Link("datacenter", "pe3", capacity_mbps=10000, base_latency_ms=2))
    topo.add_link(Link("pe1", "p1", capacity_mbps=10000, base_latency_ms=2))
    topo.add_link(Link("pe2", "p1", capacity_mbps=10000, base_latency_ms=2))
    topo.add_link(Link("p1", "p2", capacity_mbps=10000, base_latency_ms=2))
    topo.add_link(Link("pe3", "p2", capacity_mbps=10000, base_latency_ms=2))

    # --- IPSec overlay tunnels (SD-WAN) ---
    topo.add_link(Link("branch1", "datacenter", is_tunnel=True, base_latency_ms=14))
    topo.add_link(Link("branch2", "datacenter", is_tunnel=True, base_latency_ms=15))
    topo.add_link(Link("hub", "datacenter", is_tunnel=True, base_latency_ms=11))

    return topo


if __name__ == "__main__":
    topo = build_default_topology()
    print(topo.summary())
    print("\nSites:")
    for site in topo.sites():
        print(f"  {site}  -> neighbors: {topo.neighbors(site.name)}")
    print("\nTunnels (SD-WAN overlay):")
    for ln in topo.links:
        if ln.is_tunnel:
            print(f"  {ln.a} <=> {ln.b}  (~{ln.base_latency_ms} ms)")
