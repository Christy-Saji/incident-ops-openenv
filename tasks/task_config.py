"""Task configurations for realistic DevOps incident response scenarios."""

VALID_TASKS = ["easy", "medium", "hard"]

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
}
