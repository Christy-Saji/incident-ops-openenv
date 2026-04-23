"""Environment implementation for realistic DevOps incident triage."""

from typing import Dict, List, Tuple
import copy

from env.models import Action, Observation, VALID_ACTIONS
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS


class DevOpsEnv:
    def __init__(
        self,
        task: str = "easy",
        max_steps: int | None = None,
        partial_obs: bool = False,
    ):
        self.task = task
        self.max_steps = max_steps or TASK_CONFIGS[task]["max_steps"]
        self.current_step = 0
        self.partial_obs = partial_obs
        self._state: Dict = {}
        self._last_action: str = ""
        self._episode_history: List[Dict] = []
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, partial_obs: bool | None = None) -> Dict:
        config = TASK_CONFIGS.get(self.task)
        if config is None:
            raise ValueError(f"Unknown task '{self.task}'. Choose from {list(TASK_CONFIGS)}")

        if partial_obs is not None:
            self.partial_obs = partial_obs

        self.max_steps = config["max_steps"]
        self.current_step = 0
        self._last_action = ""
        self._episode_history = []

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
            self._record_step(action, reward, done)
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
        if action == self._last_action and action in {
            "post_status_update", "inspect_auth_logs", "inspect_db_metrics",
            "inspect_network_topology", "inspect_memory_profile", "inspect_disk_usage",
        }:
            penalty -= 0.02
        if info.get("error") == "incident_not_stable":
            penalty -= 0.04

        reward = max(-0.25, min(1.0, round(raw_reward + penalty, 2)))

        # Communication quality bonus
        comm_log = self._state.get("communication_log", [])
        if action == "post_status_update" and len(comm_log) >= 1:
            reward = min(1.0, reward + 0.02)

        # Efficiency bonus — resolved faster than optimal
        expected = len(TASK_CONFIGS[self.task].get("optimal_actions", []))
        if action == "resolve_incident" and self._state.get("resolved"):
            step_count = self._state["step_count"]
            if step_count <= expected:
                bonus = round(0.10 * max(0, 1 - step_count / max(expected, 1)), 2)
                reward = min(1.0, reward + bonus)
                info["efficiency_bonus"] = str(bonus)

        # Harmful-action-rate penalty (ratio, not just raw count)
        step_count = max(1, self._state["step_count"])
        harm_rate = self._state["harmful_action_count"] / step_count
        if harm_rate > 0.3:
            rate_penalty = round(min(0.05, harm_rate * 0.05), 2)
            reward = max(-0.25, reward - rate_penalty)

        if action == "resolve_incident" and not self._state["resolved"]:
            info["error"] = "incident_not_stable"

        info["score"] = f"{score:.2f}"
        info["breakdown"] = ",".join(
            f"{key}:{value:.2f}" for key, value in breakdown.items()
        )

        done = self._state["resolved"] or self.current_step >= self.max_steps
        self._record_step(action, reward, done)
        return self.state(), reward, done, info

    def state(self) -> Dict:
        """Return current observation, honouring partial-observability mode."""
        known_findings = self._state["known_findings"]
        if self.partial_obs:
            known_findings = ["[hidden — partial observability active]"]

        observation = Observation(
            task=self._state["task"],
            incident_title=self._state["incident_title"],
            customer_impact=self._state["customer_impact"],
            incident_phase=self._state["incident_phase"],
            active_alerts=self._state["active_alerts"],
            service_status=self._state["service_status"],
            metrics=self._state["metrics"],
            known_findings=known_findings,
            communication_log=self._state["communication_log"],
            recent_actions=self._state["recent_actions"],
            available_actions=VALID_ACTIONS,
            partial_observability=self.partial_obs,
        )
        return observation.model_dump()

    def get_state(self) -> Dict:
        return self.state()

    def episode(self) -> Dict:
        """Return full episode trace for replay/judging."""
        total_score, breakdown = compute_score(self.task, self._state)
        return {
            "task": self.task,
            "partial_observability": self.partial_obs,
            "total_steps": self.current_step,
            "max_steps": self.max_steps,
            "resolved": self._state.get("resolved", False),
            "final_score": round(total_score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "harmful_action_count": self._state.get("harmful_action_count", 0),
            "communication_log": self._state.get("communication_log", []),
            "trajectory": self._episode_history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_step(self, action: str, reward: float, done: bool) -> None:
        self._episode_history.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "done": done,
            "observation": self.state(),
        })

    def _apply_action(self, action: str) -> None:
        config = TASK_CONFIGS[self.task]

        if action == "acknowledge_incident":
            self._append_unique(
                self._state["communication_log"],
                "Incident acknowledged and ownership assumed.",
            )
            self._state["incident_phase"] = "investigating"
            return

        if action == "inspect_auth_logs":
            for hint in config.get("log_hints", []):
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "inspect_db_metrics":
            for hint in config.get("db_hints", []):
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "inspect_deploy_history":
            for deploy in config.get("recent_deploys", []):
                self._append_unique(
                    self._state["known_findings"], f"deploy history: {deploy}"
                )
            self._append_unique(
                self._state["known_findings"],
                f"runbook: {config.get('runbook_hint', '')}",
            )
            return

        if action == "inspect_network_topology":
            for hint in config.get("log_hints", []):
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "inspect_memory_profile":
            for hint in config.get("log_hints", []):
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "inspect_disk_usage":
            for hint in config.get("log_hints", []):
                self._append_unique(self._state["known_findings"], hint)
            return

        if action == "rollback_auth_deploy":
            if self.task in {"easy", "hard"}:
                self._state["service_status"]["auth"] = "running"
                self._append_unique(
                    self._state["known_findings"],
                    "Auth rollback completed successfully.",
                )
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "rollback_service_deploy":
            if self.task == "memory_leak":
                self._append_unique(
                    self._state["known_findings"],
                    "Payment-service rolled back to v3.2.0 — OOM kills stopped.",
                )
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "restart_auth_service":
            if self.task == "easy":
                self._append_unique(
                    self._state["known_findings"],
                    "Auth restart reduced errors briefly but root cause remains.",
                )
            else:
                self._state["harmful_action_count"] += 1
                self._append_unique(
                    self._state["known_findings"],
                    "Auth restart caused a brief thundering herd during recovery.",
                )
            return

        if action == "scale_db_cluster":
            if self.task in {"medium", "hard", "memory_leak"}:
                self._state["service_status"]["db"] = "running"
                self._append_unique(
                    self._state["known_findings"], "Database capacity increased."
                )
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "flush_cache":
            self._state["service_status"]["cache"] = "running"
            self._append_unique(
                self._state["known_findings"],
                "Cache flush reduced stale session pressure.",
            )
            if self.task == "medium":
                self._state["harmful_action_count"] += 1
            return

        if action == "shift_traffic_canary":
            self._append_unique(
                self._state["known_findings"],
                "Traffic shifted away from the hottest shard.",
            )
            return

        if action == "withdraw_bgp_route":
            if self.task == "network":
                self._append_unique(
                    self._state["known_findings"],
                    "BGP withdrawal sent — AS64500 no longer advertising leaked prefix.",
                )
                self._state["service_status"]["network"] = "running"
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "archive_old_logs":
            if self.task == "disk_full":
                self._append_unique(
                    self._state["known_findings"],
                    "Old logs compressed and archived — 14 GB freed on /var/log.",
                )
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "reduce_log_verbosity":
            if self.task == "disk_full":
                self._append_unique(
                    self._state["known_findings"],
                    "Log level reverted to INFO — log volume reduced 8×.",
                )
            else:
                self._state["harmful_action_count"] += 1
            return

        if action == "post_status_update":
            count = len(self._state["communication_log"])
            msgs = [
                "Initial status update posted: incident is under active investigation.",
                "Status page updated with mitigation progress and next review time.",
                f"Follow-up update #{count}: continued monitoring, no further customer action needed.",
            ]
            msg = msgs[min(count, len(msgs) - 1)]
            self._append_unique(self._state["communication_log"], msg)
            return

        if action == "resolve_incident":
            if self._can_resolve():
                self._state["resolved"] = True
                self._state["incident_phase"] = "resolved"
                self._append_unique(
                    self._state["communication_log"],
                    "Incident marked resolved after metrics stabilized.",
                )
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

        elif self.task == "network":
            if "withdraw_bgp_route" in actions:
                status["network"] = "running"
                metrics["latency_ms"] = min(metrics["latency_ms"], 180)
                metrics["error_rate"] = min(metrics["error_rate"], 2)
            if "shift_traffic_canary" in actions:
                metrics["latency_ms"] = min(metrics["latency_ms"], 95)
                metrics["error_rate"] = min(metrics["error_rate"], 1)
            if "withdraw_bgp_route" in actions and "shift_traffic_canary" in actions:
                status["api"] = "running"
                metrics["latency_ms"] = 88
                metrics["cpu_usage"] = 39
                metrics["error_rate"] = 1

        elif self.task == "memory_leak":
            if "rollback_service_deploy" in actions:
                metrics["memory_usage"] = min(metrics["memory_usage"], 70)
                metrics["error_rate"] = min(metrics["error_rate"], 6)
            if "scale_db_cluster" in actions:
                status["db"] = "running"
                metrics["latency_ms"] = min(metrics["latency_ms"], 200)
            if "rollback_service_deploy" in actions and "scale_db_cluster" in actions:
                status["api"] = "running"
                metrics["memory_usage"] = 55
                metrics["latency_ms"] = 118
                metrics["error_rate"] = 2
                metrics["cpu_usage"] = 48

        elif self.task == "disk_full":
            if "archive_old_logs" in actions:
                metrics["latency_ms"] = min(metrics["latency_ms"], 310)
                metrics["error_rate"] = min(metrics["error_rate"], 5)
            if "reduce_log_verbosity" in actions:
                metrics["latency_ms"] = min(metrics["latency_ms"], 140)
                metrics["error_rate"] = min(metrics["error_rate"], 2)
            if "archive_old_logs" in actions and "reduce_log_verbosity" in actions:
                status["api"] = "running"
                metrics["latency_ms"] = 125
                metrics["cpu_usage"] = 54
                metrics["error_rate"] = 1

        if self._is_stable():
            self._state["incident_phase"] = "monitoring"
            self._state["active_alerts"] = ["incident mitigated, monitoring recovery"]

    def _is_stable(self) -> bool:
        metrics = self._state["metrics"]
        status = self._state["service_status"]
        services_ok = all(v == "running" for v in status.values())
        return (
            services_ok
            and metrics["cpu_usage"] <= 65
            and metrics["memory_usage"] <= 70
            and metrics["latency_ms"] <= 140
            and metrics["error_rate"] <= 3
        )

    def _can_resolve(self) -> bool:
        required = set(TASK_CONFIGS[self.task]["good_followups"]) - {"resolve_incident"}
        completed = set(self._state["actions_taken"])
        return self._is_stable() and required.issubset(completed)

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value not in items:
            items.append(value)
