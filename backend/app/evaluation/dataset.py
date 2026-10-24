from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationSample:
    """A benchmark Q&A evaluation sample."""
    sample_id: str
    question: str
    ground_truth_answer: str
    relevant_doc_id: str
    relevant_page: int
    relevant_keywords: List[str] = field(default_factory=list)


class EvaluationDataset:
    """Phase 18: Benchmark evaluation dataset management."""

    DEFAULT_SAMPLES = [
        EvaluationSample(
            sample_id="eval_1",
            question="What is polymorphism and how does runtime polymorphism work in C++?",
            ground_truth_answer="Polymorphism allows objects of different classes to respond to the same function call. Runtime polymorphism is achieved in C++ through virtual functions, vtables, and dynamic binding.",
            relevant_doc_id="oops_notes_doc",
            relevant_page=5,
            relevant_keywords=["polymorphism", "virtual", "vtable", "dynamic"],
        ),
        EvaluationSample(
            sample_id="eval_2",
            question="What is the role of virtual destructors in C++ base classes?",
            ground_truth_answer="A virtual destructor ensures that when a derived class object is deleted via a base class pointer, the derived destructor is invoked first followed by the base destructor, preventing memory leaks.",
            relevant_doc_id="oops_notes_doc",
            relevant_page=12,
            relevant_keywords=["virtual", "destructor", "memory leak", "derived"],
        ),
        EvaluationSample(
            sample_id="eval_3",
            question="What is encapsulation and data hiding?",
            ground_truth_answer="Encapsulation bundles data members and methods together inside a class, while restricting direct access using access specifiers like private and protected to achieve data hiding.",
            relevant_doc_id="oops_notes_doc",
            relevant_page=3,
            relevant_keywords=["encapsulation", "data hiding", "access specifiers", "private"],
        ),
    ]

    def __init__(self, samples: Optional[List[EvaluationSample]] = None) -> None:
        self.samples = samples or list(self.DEFAULT_SAMPLES)

    def add_sample(self, sample: EvaluationSample) -> None:
        self.samples.append(sample)

    def get_samples(self) -> List[EvaluationSample]:
        return self.samples
