from backend.app.agents.router import AgenticQueryRouter
from backend.app.agents.validator import AnswerValidator


def test_agentic_query_router_classification():
    router = AgenticQueryRouter()

    # Text route
    text_res = router.classify_query("What is runtime polymorphism in C++?")
    assert text_res["route"] == "TEXT"

    # Image route
    img_res = router.classify_query("Explain the architecture diagram in Figure 4.")
    assert img_res["route"] == "IMAGE"

    # Table route
    tbl_res = router.classify_query("Which model has the highest F1 score benchmark?")
    assert tbl_res["route"] == "TABLE"

    # Graph route
    graph_res = router.classify_query("How are virtual functions and vtable related?")
    assert graph_res["route"] == "GRAPH"

    # Hybrid route
    hybrid_res = router.classify_query("Compare the accuracy metrics table and Figure 2 diagram.")
    assert hybrid_res["route"] == "HYBRID"


def test_answer_validator_grounding():
    validator = AnswerValidator()

    context = (
        "Virtual functions enable runtime polymorphism in C++. "
        "They utilize a virtual method table (vtable) and a pointer (vptr) "
        "to dispatch dynamic method calls at execution time."
    )

    grounded_answer = (
        "In C++, virtual functions achieve runtime polymorphism through a vtable and vptr. "
        "This dynamic dispatch mechanism resolves method calls at execution time."
    )

    res = validator.validate_answer(
        answer=grounded_answer,
        context=context,
        citations=[{"document_id": "doc_1", "page_number": 5}],
    )

    assert res["is_valid"] is True
    assert res["grounding_score"] >= 0.7
    assert res["needs_re_retrieval"] is False
    assert res["citation_count"] == 1


def test_answer_validator_detects_hallucinations():
    validator = AnswerValidator()

    context = "PyTorch is an open source machine learning framework based on the Torch library."
    hallucinated_answer = "Quantum entanglement teleportation was achieved using superconducting transmon qubits."

    res = validator.validate_answer(answer=hallucinated_answer, context=context)

    assert res["is_valid"] is False
    assert res["grounding_score"] < 0.4
    assert res["needs_re_retrieval"] is True
    assert len(res["ungrounded_claims"]) > 0
