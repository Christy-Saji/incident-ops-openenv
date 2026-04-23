"""Environment implementation for realistic DevOps incident triage."""

from typing import Dict, Tuple
import copy

from env.models import Action, Observation, VALID_ACTIONS
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS


class DevOpsEnv:
    def __init__(self, task: str = "easy", max_steps: int | None = None):
        self.task = task
        self.max_steps = max_steps or TASK_CONFIGS[task]["max_steps"]
        self.current_step = 0
        self._state: Dict = {}
        self._last_action: str = ""

    def reset(self) -> Dict:
        config = TASK_CONFIGS.get(self.task)
        if config is None:
            raise ValueError(f"Unknown task '{self.task}'. Choose from {list(TASK_CONFIGS)}")

        self.max_steps = config["max_steps"]
        self.current_step = 0
        self._last_action = ""

        self._state = {
            "task": self.task,
            "incident_title": config["title"],
            "customer_impact": config["customer_impact"],
            "incident_phase": "investigating",
            "service_status": copy.deepcopy(config["service_status"]),
            "metrics": copy.deepcopy(config["metrics"]),
            "active_alerts": list(config["alerts"]),
            "known_findings": [],
            "communication_log": [],
            "recent_actions": [],
            "actions_taken": [],
            "resolved": False,
            "harmful_action_count": 0,
            "step_count": 0,
        }
        return self.state()

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        info: Dict = {}
        previous_score, _ = compute_score(self.task, self._state)

        if action not in VALID_ACTIONS:
            self.current_step += 1
            self._state["step_count"] = self.current_step
            self._state["harmful_action_count"] += 1
            info["error"] = "invalid_action"
            reward = -0.10
            done = self.current_step >= self.max_steps
            return self.state(), reward, done, info

        Action(name=action)

        self._state["actions_taken"].append(action)
        self._state["recent_actions"] = (self._state["recent_actions"] + [action])[-5:]

        if action == self._last_action and action in {"restart_auth_service", "no_op"}:
            self._state["harmful_action_count"] += 1
        self._last_action = action

        self._apply_action(action)
        self._apply_background_dynamics()

        self.current_step += 1
        self._state["step_count"] = self.current_step

        score, breakdown = compute_score(self.task, self._state)
        raw_reward = round(score - previous_score, 2)
        penalty = 0.0

        if action == "no_op":
            penalty -= 0.03
        if action == self._last_action and action in {"post_status_update", "inspect_auth_logs", "inspect_db_metrics"}:
            penalty -= 0.02
        if info.get("error") == "incident_not_stable":
            penalty -= 0.04

        reward = max(-0.25, min(1.0, round(raw_reward + penalty, 2)))

        if action == "resolve_incident" and not self._state["resolved"]:
            info["error"] = "incident_not_stable"

        info["score"] = f"{score:.2f}"
        info["breakdown"] = ",".join(f"{key}:{value:.2f}" for key, value in breakdown.items())

        done = self._state["resolved"] or self.current_step >= self.max_steps
        return self.state(), reward, done, info

    def state(self) -> Dict:
        observation = Observation(
            task=self._state["task"],
            incident_title=self._state["incident_title"],
            customer_impact=self._state["customer_impact"],
            incident_phase=self._state["incident_phase"],
            active_alerts=self._state["active_alerts"],
            service_status=self._state["service_status"],
            metrics=self._state["metrics"],
            known_findings=self._state["known_findings"],
            communication_log=self._state["communication_log"],
            recent_actions=self._state["recent_actions"],
            available_actions=VALID_ACTIONS,
        )
        return observation.model_dump()

    def get_state(self) -> Dict:
        return self.state()

    def _apply_action(self, action: str) -> None:
        config = TASK_CONFIGS[self.task]

        if action == "acknowledge_incident":
            self._append_unique(self._state["communication_log"], "Incident acknowledged and ownership assumed.")
            self._state["incident_phase"] = "investigating"
            return

        if action == "inspect_auth_logs":
            for hint in config["log_hints"]:
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "inspect_db_metrics":
            for hint in config["db_hints"]:
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "inspect_deploy_history":
            for deploy in config["recent_deploys"]:
                self._append_unique(self._state["known_findings"], f"deploy history: {deploy}")
            self._append_unique(self._state["known_findings"], f"runbook: {config['runbook_hint']}")
            return

        if action == "rollback_auth_deploy":
            if self.task in {"easy", "hard"}:
                self._state["service_status"]["auth"] = "running"
                self._append_unique(self._state["known_findings"], "Auth rollback completed successfully.")
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "restart_auth_service":
            if self.task == "easy":
                self._append_unique(self._state["known_findings"], "Auth restart reduced errors briefly but root cause remains.")
            else:
                self._state["harmful_action_count"] += 1
                self._append_unique(self._state["known_findings"], "Auth restart caused a brief thundering herd during recovery.")
            return

        if action == "scale_db_cluster":
            if self.task in {"medium", "hard"}:
                self._state["service_status"]["db"] = "running"
                self._append_unique(self._state["known_findings"], "Database capacity increased.")
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "flush_cache":
            self._state["service_status"]["cache"] = "running"
            self._append_unique(self._state["known_findings"], "Cache flush reduced stale session pressure.")
            if self.task == "medium":
                self._state["harmful_action_count"] += 1
            return

        if action == "shift_traffic_canary":
            self._append_unique(self._state["known_findings"], "Traffic shifted away from the hottest shard.")
            return

        if action == "post_status_update":
            self._append_unique(
                self._state["communication_log"],
                "Status page updated with mitigation progress and next review time.",
            )
            return

        if action == "resolve_incident":
            if self._can_resolve():
                self._state["resolved"] = True
                self._state["incident_phase"] = "resolved"
                self._append_unique(self._state["communication_log"], "Incident marked resolved after metrics stabilized.")
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "no_op":
            self._state["harmful_action_count"] += 1

    def _apply_background_dynamics(self) -> None:
        actions = set(self._state["actions_taken"])
        metrics = self._state["metrics"]
        status = self._state["service_status"]

        if self.task == "easy":
            if "rollback_auth_deploy" in actions:
                status["auth"] = "running"
                status["api"] = "running"
                metrics["latency_ms"] = 125
                metrics["error_rate"] = 2
                metrics["cpu_usage"] = 48
            elif "restart_auth_service" in actions:
                status["auth"] = "degraded"
                status["api"] = "degraded"
                metrics["latency_ms"] = 185
                metrics["error_rate"] = 12

        elif self.task == "medium":
            status["auth"] = "running"
            if "scale_db_cluster" in actions:
                status["db"] = "running"
                metrics["latency_ms"] = 240
                metrics["cpu_usage"] = 67
                metrics["error_rate"] = 6
            if "shift_traffic_canary" in actions:
                metrics["request_rate"] = 1120
                metrics["latency_ms"] = min(metrics["latency_ms"], 150)
                metrics["error_rate"] = min(metrics["error_rate"], 3)
                status["api"] = "running" if "scale_db_cluster" in actions else "degraded"
            if "scale_db_cluster" in actions and "shift_traffic_canary" in actions:
                status["api"] = "running"
                status["cache"] = "running"
                metrics["cpu_usage"] = 58
                metrics["memory_usage"] = 63
                metrics["latency_ms"] = 115
                metrics["error_rate"] = 1

        elif self.task == "hard":
            if "rollback_auth_deploy" in actions:
                status["auth"] = "running"
                status["api"] = "degraded"
                metrics["error_rate"] = 14
                metrics["latency_ms"] = 480
            if "scale_db_cluster" in actions:
                status["db"] = "running"
                metrics["cpu_usage"] = 70
                metrics["latency_ms"] = min(metrics["latency_ms"], 290)
            if "shift_traffic_canary" in actions:
                metrics["request_rate"] = 1180
                metrics["cpu_usage"] = min(metrics["cpu_usage"], 61)
                metrics["latency_ms"] = min(metrics["latency_ms"], 150)
                metrics["error_rate"] = min(metrics["error_rate"], 4)
            if "flush_cache" in actions:
                status["cache"] = "running"
                metrics["memory_usage"] = min(metrics["memory_usage"], 67)
            if {"rollback_auth_deploy", "scale_db_cluster", "shift_traffic_canary"}.issubset(actions):
                status["auth"] = "running"
                status["api"] = "running"
                status["db"] = "running"
                status["cache"] = "running" if "flush_cache" in actions else "degraded"
                metrics["cpu_usage"] = 59
                metrics["memory_usage"] = 69 if "flush_cache" not in actions else 63
                metrics["latency_ms"] = 125
                metrics["error_rate"] = 2

        if self._is_stable():
            self._state["incident_phase"] = "monitoring"
            self._state["active_alerts"] = ["incident mitigated, monitoring recovery"]

    def _is_stable(self) -> bool:
        metrics = self._state["metrics"]
        status = self._state["service_status"]
        services_ok = all(value == "running" for value in status.values())
        return (
            services_ok
            and metrics["cpu_usage"] <= 65
            and metrics["memory_usage"] <= 70
            and metrics["latency_ms"] <= 140
            and metrics["error_rate"] <= 3
        )

    def _can_resolve(self) -> bool:
        required_followups = set(TASK_CONFIGS[self.task]["good_followups"]) - {"resolve_incident"}
        completed = set(self._state["actions_taken"])
        return self._is_stable() and required_followups.issubset(completed)

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value not in items:
            items.append(value)
