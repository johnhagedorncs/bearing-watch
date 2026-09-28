"""Structural checks on the deployment manifests.

These cannot verify that the manifests WORK -- that needs a cluster. They verify
the properties that are easy to regress silently and expensive to discover in
production: that the scheduled workflow ships suspended, that workflow bodies
are referenced rather than inlined, that containers do not run as root, and that
liveness and readiness are actually different endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ARGO_DIR = REPO_ROOT / "deploy" / "argo"
K8S_DIR = REPO_ROOT / "deploy" / "k8s"


def load_all(path: Path) -> list[dict[str, Any]]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def load_one(path: Path) -> dict[str, Any]:
    docs = load_all(path)
    assert len(docs) == 1, f"{path} should hold exactly one document"
    return docs[0]


def all_yaml() -> list[Path]:
    return sorted(
        [*ARGO_DIR.rglob("*.yaml"), *K8S_DIR.rglob("*.yaml"), REPO_ROOT / "docker-compose.yml"]
    )


@pytest.mark.parametrize("path", all_yaml(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_manifest_is_valid_yaml(path: Path) -> None:
    docs = load_all(path)
    assert docs, f"{path} parsed to nothing"


@pytest.mark.parametrize(
    "path",
    sorted([*ARGO_DIR.rglob("*.yaml"), *K8S_DIR.rglob("*.yaml")]),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_every_k8s_document_has_kind_and_name(path: Path) -> None:
    for doc in load_all(path):
        if doc.get("kind") == "Component":
            continue  # kustomize components have no metadata.name
        if "kustomize" in str(doc.get("apiVersion", "")):
            continue
        assert doc.get("apiVersion"), f"{path}: document without apiVersion"
        assert doc.get("kind"), f"{path}: document without kind"
        assert doc.get("metadata", {}).get("name"), f"{path}: {doc['kind']} without a name"


# ------------------------------------------------------------------- cron


def test_cronworkflow_is_suspended_in_the_base() -> None:
    """The single most important property in deploy/.

    An active-on-apply CronWorkflow starts running the moment it lands in any
    cluster it is synced to, including one where the data mount is wrong. It
    does not error -- it runs, and writes results that look real.
    """
    cron = load_one(ARGO_DIR / "cronworkflow-score.yaml")
    assert cron["kind"] == "CronWorkflow"
    assert cron["spec"]["suspend"] is True, (
        "the base CronWorkflow must ship SUSPENDED; enabling it is a per-environment "
        "decision made by the enable-cron component"
    )


def test_cronworkflow_references_a_template_instead_of_inlining_it() -> None:
    """Inlined bodies drift from the referenced ones, and nobody diffs them."""
    cron = load_one(ARGO_DIR / "cronworkflow-score.yaml")
    spec = cron["spec"]["workflowSpec"]
    assert "workflowTemplateRef" in spec, "the workflow body must be referenced by name"
    assert spec["workflowTemplateRef"]["name"] == "bearing-watch-score"
    assert "templates" not in spec, "the CronWorkflow must not inline template bodies"


def test_cronworkflow_forbids_concurrent_runs() -> None:
    cron = load_one(ARGO_DIR / "cronworkflow-score.yaml")
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"


def test_enable_cron_component_flips_suspend_to_false() -> None:
    component = load_one(K8S_DIR / "components" / "enable-cron" / "kustomization.yaml")
    assert component["kind"] == "Component"
    body = yaml.safe_load(component["patches"][0]["patch"])
    assert body[0]["path"] == "/spec/suspend"
    assert body[0]["value"] is False


@pytest.mark.parametrize("environment", ["dev", "prod"])
def test_overlays_opt_into_the_schedule_explicitly(environment: str) -> None:
    overlay = load_one(K8S_DIR / "overlays" / environment / "kustomization.yaml")
    assert "../../components/enable-cron" in overlay.get("components", []), (
        f"the {environment} overlay must explicitly opt in to the schedule"
    )


# -------------------------------------------------------------- templates


@pytest.mark.parametrize("name", ["workflowtemplate-train.yaml", "workflowtemplate-score.yaml"])
def test_workflow_templates_are_named_and_have_an_entrypoint(name: str) -> None:
    template = load_one(ARGO_DIR / name)
    assert template["kind"] == "WorkflowTemplate"
    assert template["spec"]["entrypoint"]
    assert template["spec"]["serviceAccountName"] == "bearing-watch"
    entrypoint = template["spec"]["entrypoint"]
    assert any(t["name"] == entrypoint for t in template["spec"]["templates"])


@pytest.mark.parametrize("name", ["workflowtemplate-train.yaml", "workflowtemplate-score.yaml"])
def test_workflow_templates_have_a_deadline(name: str) -> None:
    """A pod stuck on a network mount holds its resources until someone notices."""
    template = load_one(ARGO_DIR / name)
    assert template["spec"]["activeDeadlineSeconds"] > 0


def test_train_and_score_are_separate_templates() -> None:
    """Different schedules, resources, retries and failure consequences."""
    train = load_one(ARGO_DIR / "workflowtemplate-train.yaml")
    score = load_one(ARGO_DIR / "workflowtemplate-score.yaml")
    assert train["metadata"]["name"] != score["metadata"]["name"]


def test_score_workflow_runs_preflight_before_inference() -> None:
    """A data problem must fail before compute is spent, under its own name."""
    score = load_one(ARGO_DIR / "workflowtemplate-score.yaml")
    templates = {t["name"]: t for t in score["spec"]["templates"]}
    assert "preflight" in templates
    steps = templates["score"]["steps"]
    order = [s[0]["name"] for s in steps]
    assert order.index("preflight") < order.index("infer")


def test_run_id_has_no_latest_default() -> None:
    """'Whatever trained most recently' is not a model version.

    Resolving it at runtime changes behaviour with no deployment and nothing in
    git to show for it.
    """
    score = load_one(ARGO_DIR / "workflowtemplate-score.yaml")
    params = {p["name"]: p.get("value") for p in score["spec"]["arguments"]["parameters"]}
    assert params["run-id"] in ("", None), "run-id must not default to a resolvable value"


def _containers_in(doc: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(doc, dict):
        looks_like_container = "image" in doc and (
            "command" in doc or "args" in doc or "name" in doc
        )
        if looks_like_container and ("securityContext" in doc or "resources" in doc):
            found.append(doc)
        for value in doc.values():
            found.extend(_containers_in(value))
    elif isinstance(doc, list):
        for item in doc:
            found.extend(_containers_in(item))
    return found


@pytest.mark.parametrize(
    "name",
    ["workflowtemplate-train.yaml", "workflowtemplate-score.yaml"],
)
def test_workflow_containers_are_hardened(name: str) -> None:
    """A container running as root has root on any host path it is handed."""
    doc = load_one(ARGO_DIR / name)
    containers = _containers_in(doc)
    assert containers, f"no containers found in {name}"
    for container in containers:
        sc = container.get("securityContext", {})
        assert sc.get("runAsNonRoot") is True, f"{name}: container may run as root"
        assert sc.get("allowPrivilegeEscalation") is False
        assert sc.get("readOnlyRootFilesystem") is True
        assert sc.get("capabilities", {}).get("drop") == ["ALL"]
        assert container.get("resources", {}).get("requests"), "missing resource requests"


# ------------------------------------------------------------- deployment


def test_api_deployment_separates_liveness_from_readiness() -> None:
    """Liveness restarts a wedged process; readiness gates traffic.

    A liveness probe pointed at /ready restarts a pod whose model is missing,
    which does not produce a model and does hide the problem behind a crash loop.
    """
    deployment = load_one(K8S_DIR / "base" / "deployment-api.yaml")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert (
        container["livenessProbe"]["httpGet"]["path"]
        != container["readinessProbe"]["httpGet"]["path"]
    )


def test_api_deployment_runs_as_non_root() -> None:
    deployment = load_one(K8S_DIR / "base" / "deployment-api.yaml")
    pod_sc = deployment["spec"]["template"]["spec"]["securityContext"]
    assert pod_sc["runAsNonRoot"] is True
    assert pod_sc["runAsUser"] == 10001

    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False


def test_api_deployment_pins_an_explicit_run_id() -> None:
    """Resolving 'latest' at pod start lets a rolling restart change the model."""
    deployment = load_one(K8S_DIR / "base" / "deployment-api.yaml")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["BEARINGWATCH_RUN_DIR"].startswith("/runs/")
    assert "latest" not in env["BEARINGWATCH_RUN_DIR"].lower()


def test_api_mounts_run_folders_read_only() -> None:
    """The serving pod has no business writing to the artifacts it serves."""
    deployment = load_one(K8S_DIR / "base" / "deployment-api.yaml")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    runs = next(m for m in container["volumeMounts"] if m["name"] == "runs")
    assert runs["readOnly"] is True


def test_prod_overlay_pins_an_image_digest_not_a_tag() -> None:
    """A tag is a mutable pointer; a repushed tag changes code with no git diff."""
    overlay = load_one(K8S_DIR / "overlays" / "prod" / "kustomization.yaml")
    image = overlay["images"][0]
    assert "digest" in image, "prod must pin by digest"
    assert "newTag" not in image


def test_base_kustomization_includes_every_argo_manifest() -> None:
    base = load_one(K8S_DIR / "base" / "kustomization.yaml")
    referenced = {Path(r).name for r in base["resources"]}
    for path in sorted(ARGO_DIR.glob("*.yaml")):
        assert path.name in referenced, f"{path.name} is not referenced by the base"


# ---------------------------------------------------------- docker compose


def test_compose_brings_up_api_mlflow_and_monitor() -> None:
    compose = load_one(REPO_ROOT / "docker-compose.yml")
    assert set(compose["services"]) >= {"api", "mlflow", "monitor"}


def test_compose_mounts_runs_read_only_for_the_api() -> None:
    compose = load_one(REPO_ROOT / "docker-compose.yml")
    assert any(v.endswith(":ro") for v in compose["services"]["api"]["volumes"])


def test_dockerfile_is_multistage_and_non_root() -> None:
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert text.count("FROM ") >= 2, "the Dockerfile must be multi-stage"
    assert "AS builder" in text
    assert "USER bearing" in text, "the runtime image must not run as root"
    assert "HEALTHCHECK" in text
    # The runtime stage must not install dev dependencies.
    runtime = text.split("AS runtime", 1)[1]
    for forbidden in ("pytest", "ruff", "mypy", "build-essential"):
        assert forbidden not in runtime, f"{forbidden} leaked into the runtime image"
