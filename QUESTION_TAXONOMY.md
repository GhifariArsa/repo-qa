## Question tiers

| Tier | Name                                                                | Templates                                                                                 |
| ---- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1    | existence / lookup                                                  | entity type, method owner, defining file, direct base, has field                          |
| 2    | path traversal                                                      | path exists, shortest chain, hop count, reachable-within-2, reaches-global-read           |
| 3    | aggregation over shape                                              | fan-in, fan-out, reachability-set size, nearest common ancestor, max fan-in, method count |
| 4    | counterfactual / negative                                           | absent entity, no call path, wrong method owner, no callers, not-inherits                 |
| 5    | cross-graph composition (still thinking if this should be included) | CALLS closure × USES — **reduced scope, off by default** (see limitations)                |

### Tier 1: existence / lookup

- t1_entity_type — what kind of entity is X (class, function, method, field)
- t1_method_owner — which class defines a given method
- t1_defining_file — which file a given entity is defined in
- t1_direct_base — what class(es) does X directly inherit from
- t1_has_field — does class X have field Y (boolean)

### Tier 2: Path Traversal

- `t2_path_exists` — is there any call path from X to Y (boolean)
- `t2_shortest_chain` — what is the shortest call path from X to Y, listing every node on it
- `t2_hop_count` — how many hops is the shortest path from X to Y
- `t2_reachable_within_2` — what's reachable from X within 2 call-hops
- `t2_reaches_global_read` — does any call chain from X eventually read global variable G (composes CALLS with USES — the one template in this tier that reasons across two edge types, not just CALLS)

### Tier 3: Aggregation Over Graph Shape

- `t3_fan_in` — how many distinct callers call X directly
- `t3_fan_out` — how many distinct functions/methods does X call directly
- `t3_reach_set_size` — how many unique callables are reachable (any depth) from X
- `t3_common_ancestor` — do classes X and Y share a common ancestor via INHERITS
- `t3_max_fan_in` — which node(s) have the single highest fan-in in the repo
- `t3_class_method_count` — how many methods are defined directly on class X (excluding inherited)

### Tier 4 — Counterfactual / Negative

- `t4_no_callers` — confirm that X has no incoming callers
- `t4_absent_entity` — confirm that a plausible-sounding entity does not exist
- `t4_no_call_path` — confirm there is no call path from X to Y
- `t4_not_inherits` — confirm X does not inherit from Y
- `t4_wrong_method_owner` — confirm a method name is not defined or inherited on a given class
