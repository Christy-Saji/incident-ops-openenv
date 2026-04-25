"""Task configurations for realistic DevOps incident response scenarios.

Hint key routing (environment.py dispatches each inspect action to its own key):
  inspect_auth_logs        → log_hints
  inspect_db_metrics       → db_hints
  inspect_deploy_history   → recent_deploys + runbook_hint
  inspect_network_topology → network_hints
  inspect_memory_profile   → memory_hints
  inspect_disk_usage       → disk_hints

For tasks where a particular inspect is irrelevant, the hint key contains a
'not the root cause' message so the agent learns not to waste steps.
"""

VALID_TASKS = ["easy", "medium", "hard", "network", "memory_leak", "disk_full"]

TASK_CONFIGS = {
    "easy": {
        "title": "Auth deploy regression",
        "description": "A new auth rollout is causing login failures for new sessions.",
        "max_steps": 8,
        "customer_impact": "35% of new login attempts fail in the last 10 minutes.",
        "service_status": {
            "auth": "degraded",
            "api": "degraded",
            "db": "running",
            "cache": "running",
        },
        "metrics": {
            "cpu_usage": 58,
            "memory_usage": 62,
            "latency_ms": 210,
            "error_rate": 18,
            "request_rate": 520,
        },
        "alerts": [
            "auth 5xx rate above SLO",
            "login success rate below 70%",
        ],
        "recent_deploys": [
            "auth-service deployed version 2026.04.08.1 twelve minutes ago",
        ],
        "log_hints": [
            "auth-service log: NullReferenceException in login token validation",
            "auth-service log: error rate jumped immediately after deploy",
        ],
        "db_hints": [
            "db metrics: healthy p95 latency, normal connection count",
        ],
        "network_hints": [
            "network log: no BGP anomalies detected — network layer is healthy",
        ],
        "memory_hints": [
            "memory profile: heap usage normal — no OOM kills detected",
        ],
        "disk_hints": [
            "disk usage: /var/log at 18% — log rotation is healthy",
        ],
        "runbook_hint": "Bad deploys should usually be mitigated by rollback before restart.",
        "required_diagnostics": ["inspect_deploy_history"],
        "required_mitigations": ["rollback_auth_deploy"],
        "good_followups": ["acknowledge_incident", "resolve_incident"],
        "optimal_actions": [
            "acknowledge_incident",
            "inspect_deploy_history",
            "rollback_auth_deploy",
            "resolve_incident",
        ],
    },
    "medium": {
        "title": "Database saturation from traffic spike",
        "description": "A flash sale has pushed the primary database into saturation.",
        "max_steps": 10,
        "customer_impact": "Checkout latency is above 800ms and intermittent timeouts are reported.",
        "service_status": {
            "auth": "running",
            "api": "degraded",
            "db": "degraded",
            "cache": "running",
        },
        "metrics": {
            "cpu_usage": 78,
            "memory_usage": 68,
            "latency_ms": 420,
            "error_rate": 11,
            "request_rate": 1450,
        },
        "alerts": [
            "database cpu above 85%",
            "checkout p95 latency above 400ms",
        ],
        "recent_deploys": [
            "no production deploys in the last 24 hours",
        ],
        "log_hints": [
            "api log: upstream db timeout during checkout flow",
        ],
        "db_hints": [
            "db metrics: cpu pegged at 92%, read replica lag increasing",
            "db metrics: no schema migration running, saturation tracks request volume",
        ],
        "network_hints": [
            "network log: no routing anomalies — network layer is healthy",
        ],
        "memory_hints": [
            "memory profile: all services within normal heap bounds",
        ],
        "disk_hints": [
            "disk usage: /var/log at 22% — log rotation is healthy",
        ],
        "runbook_hint": "Traffic spikes are usually mitigated by scaling the DB and shifting traffic.",
        "required_diagnostics": ["inspect_db_metrics"],
        "required_mitigations": ["scale_db_cluster", "shift_traffic_canary"],
        "good_followups": ["acknowledge_incident", "post_status_update", "resolve_incident"],
        "optimal_actions": [
            "acknowledge_incident",
            "inspect_db_metrics",
            "scale_db_cluster",
            "shift_traffic_canary",
            "post_status_update",
            "resolve_incident",
        ],
    },
    "hard": {
        "title": "Cascading outage after auth deploy",
        "description": "A bad auth deploy caused retry storms, cache churn, and DB saturation.",
        "max_steps": 12,
        "customer_impact": "Logins fail, checkout errors are rising, and support tickets are spiking.",
        "service_status": {
            "auth": "degraded",
            "api": "degraded",
            "db": "degraded",
            "cache": "degraded",
        },
        "metrics": {
            "cpu_usage": 88,
            "memory_usage": 81,
            "latency_ms": 760,
            "error_rate": 23,
            "request_rate": 1680,
        },
        "alerts": [
            "auth error budget burn critical",
            "database saturation critical",
            "cache evictions above threshold",
        ],
        "recent_deploys": [
            "auth-service deployed version 2026.04.08.3 eighteen minutes ago",
            "no database deploys in the last 48 hours",
        ],
        "log_hints": [
            "auth-service log: token parsing panic introduced in latest release",
            "api log: repeated auth retries amplify downstream load",
        ],
        "db_hints": [
            "db metrics: cpu 96%, connection queue backed up",
            "db metrics: saturation appears secondary to auth retry storm",
        ],
        "network_hints": [
            "network log: no routing anomalies — network layer is healthy",
        ],
        "memory_hints": [
            "memory profile: heap usage within bounds — not contributing to outage",
        ],
        "disk_hints": [
            "disk usage: /var/log at 31% — log rotation is healthy",
        ],
        "runbook_hint": "For retry storms, rollback the bad deploy first, then reduce downstream pressure.",
        "required_diagnostics": [
            "inspect_auth_logs",
            "inspect_deploy_history",
            "inspect_db_metrics",
        ],
        "required_mitigations": [
            "rollback_auth_deploy",
            "scale_db_cluster",
            "shift_traffic_canary",
        ],
        "good_followups": [
            "acknowledge_incident",
            "post_status_update",
            "resolve_incident",
        ],
        "optimal_actions": [
            "acknowledge_incident",
            "inspect_auth_logs",
            "inspect_deploy_history",
            "inspect_db_metrics",
            "rollback_auth_deploy",
            "scale_db_cluster",
            "shift_traffic_canary",
            "post_status_update",
            "resolve_incident",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # NEW TASK SCENARIOS
    # ──────────────────────────────────────────────────────────────────────────

    "network": {
        "title": "BGP route leak causing latency spike",
        "description": (
            "A BGP route leak from an upstream provider has injected a longer AS path, "
            "pushing inter-region latency above SLO. Traffic is still flowing but the "
            "added hops are degrading API response times globally."
        ),
        "max_steps": 10,
        "customer_impact": (
            "API latency has doubled globally. EU users report 900ms+ response times. "
            "No service errors yet but SLO burn rate is accelerating."
        ),
        "service_status": {
            "auth": "running",
            "api": "degraded",
            "db": "running",
            "network": "degraded",
        },
        "metrics": {
            "cpu_usage": 42,
            "memory_usage": 51,
            "latency_ms": 940,
            "error_rate": 3,
            "request_rate": 1100,
        },
        "alerts": [
            "global p99 latency above 800ms SLO",
            "BGP route table anomaly detected on edge router eu-west-2",
            "cross-region RTT increased by 380ms",
        ],
        "recent_deploys": [
            "no application deploys in the last 6 hours",
            "network config pushed to edge routers 25 minutes ago",
        ],
        "log_hints": [
            "auth log: no auth deploy or errors detected — auth service is not the root cause",
        ],
        "db_hints": [
            "db metrics: nominal latency, no connection pressure — db is not the cause",
        ],
        "network_hints": [
            "network log: unexpected AS path prepend from upstream peer AS64500",
            "network log: eu-west-2 edge advertising /21 prefix via longer path since 14:02 UTC",
            "network log: traffic to eu-west-2 traversing 4 extra hops through AS64500",
        ],
        "memory_hints": [
            "memory profile: all pods within normal memory bounds — not the root cause",
        ],
        "disk_hints": [
            "disk usage: /var/log at 14% — log rotation is healthy",
        ],
        "runbook_hint": (
            "BGP route leaks are mitigated by withdrawing the leaked route and failing over "
            "traffic to healthy regions. Filtering bogon routes prevents recurrence."
        ),
        "required_diagnostics": ["inspect_network_topology", "inspect_deploy_history"],
        "required_mitigations": ["withdraw_bgp_route", "shift_traffic_canary"],
        "good_followups": ["acknowledge_incident", "post_status_update", "resolve_incident"],
        "optimal_actions": [
            "acknowledge_incident",
            "inspect_network_topology",
            "inspect_deploy_history",
            "withdraw_bgp_route",
            "shift_traffic_canary",
            "post_status_update",
            "resolve_incident",
        ],
    },

    "memory_leak": {
        "title": "OOM kills causing cascading service restarts",
        "description": (
            "A memory leak introduced in the payment-service v3.2.1 is causing OOM kills "
            "every 8–10 minutes. Each OOM kill triggers a Kubernetes pod restart which "
            "creates a thundering herd on the DB connection pool."
        ),
        "max_steps": 11,
        "customer_impact": (
            "Payment processing fails intermittently during pod restarts. "
            "10% of transactions in the past 20 minutes returned 503 or timed out."
        ),
        "service_status": {
            "auth": "running",
            "api": "degraded",
            "db": "degraded",
            "cache": "running",
        },
        "metrics": {
            "cpu_usage": 55,
            "memory_usage": 94,
            "latency_ms": 380,
            "error_rate": 14,
            "request_rate": 890,
        },
        "alerts": [
            "payment-service OOMKilled 3 times in last 30 minutes",
            "pod restart loop detected in payment namespace",
            "db connection pool exhaustion during pod cold-start",
        ],
        "recent_deploys": [
            "payment-service deployed version 3.2.1 forty minutes ago",
            "no infra changes in the last 24 hours",
        ],
        "log_hints": [
            "auth log: auth service stable, no errors — auth is not the root cause",
        ],
        "db_hints": [
            "db metrics: connection spike every 8 minutes correlates with pod restart timing",
            "db metrics: max_connections reached during payment-service cold-start",
        ],
        "network_hints": [
            "network log: no BGP or routing anomalies — network layer is healthy",
        ],
        "memory_hints": [
            "payment-service log: heap allocation growing unbounded in transaction cache",
            "payment-service log: OOMKilled signal received — memory limit 512Mi exceeded",
            "k8s log: pod payment-service-7f9d restarted 3 times, back-off delay active",
        ],
        "disk_hints": [
            "disk usage: /var/log at 19% — log rotation is healthy",
        ],
        "runbook_hint": (
            "OOM loops from a bad deploy should be stopped by rolling back the release. "
            "Temporarily increasing memory limits buys time but is not a fix. "
            "Scale the DB connection pool to absorb cold-start spikes."
        ),
        "required_diagnostics": ["inspect_memory_profile", "inspect_deploy_history"],
        "required_mitigations": ["rollback_service_deploy", "scale_db_cluster"],
        "good_followups": ["acknowledge_incident", "post_status_update", "resolve_incident"],
        "optimal_actions": [
            "acknowledge_incident",
            "inspect_memory_profile",
            "inspect_deploy_history",
            "rollback_service_deploy",
            "scale_db_cluster",
            "post_status_update",
            "resolve_incident",
        ],
    },

    "disk_full": {
        "title": "Log disk saturation blocking writes",
        "description": (
            "The /var/log partition on the primary API hosts has reached 98% capacity. "
            "Log rotation failed silently three hours ago. Application writes are now "
            "blocking because the logging library uses synchronous writes."
        ),
        "max_steps": 9,
        "customer_impact": (
            "API response times have increased by 4–6× as write threads block on disk I/O. "
            "5% of requests are timing out. New feature uploads are failing silently."
        ),
        "service_status": {
            "auth": "running",
            "api": "degraded",
            "db": "running",
            "cache": "running",
        },
        "metrics": {
            "cpu_usage": 61,
            "memory_usage": 58,
            "latency_ms": 620,
            "error_rate": 9,
            "request_rate": 740,
        },
        "alerts": [
            "disk usage /var/log above 95% on api-host-01, api-host-02",
            "log rotation cron job failed — exit code 1 — last success 3 hours ago",
            "api p95 write latency above 500ms",
        ],
        "recent_deploys": [
            "no application deploys in the last 12 hours",
            "logging verbosity increased to DEBUG level 4 hours ago for troubleshooting",
        ],
        "log_hints": [
            "auth log: auth service nominal, no errors — auth is not the root cause",
        ],
        "db_hints": [
            "db metrics: nominal — not involved in this incident",
        ],
        "network_hints": [
            "network log: no routing anomalies — network layer is healthy",
        ],
        "memory_hints": [
            "memory profile: all services within normal heap bounds — not the root cause",
        ],
        "disk_hints": [
            "system log: logrotate error — disk full, cannot rename /var/log/api/current.log",
            "system log: DEBUG logging enabled at 10:15 UTC — log volume increased 8×",
            "system log: /var/log at 98.3% — 420MB free of 20GB partition",
        ],
        "runbook_hint": (
            "Disk saturation from logs is mitigated by archiving old logs, "
            "reducing log verbosity, and fixing log rotation. "
            "Do not delete logs without archiving — required for audit."
        ),
        "required_diagnostics": ["inspect_disk_usage", "inspect_deploy_history"],
        "required_mitigations": ["archive_old_logs", "reduce_log_verbosity"],
        "good_followups": ["acknowledge_incident", "post_status_update", "resolve_incident"],
        "optimal_actions": [
            "acknowledge_incident",
            "inspect_disk_usage",
            "inspect_deploy_history",
            "archive_old_logs",
            "reduce_log_verbosity",
            "post_status_update",
            "resolve_incident",
        ],
    },
}
