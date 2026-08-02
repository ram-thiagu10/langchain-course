from dotenv import load_dotenv

load_dotenv()
from graph.chains.retrieval_grader import (
    GradeDocuments,
    parse_grade_documents,
    retrieval_grader,
)
from ingestion import retriever
from pprint import pprint
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination

from graph.chains.generation import generation_chain


def test_parse_grade_documents_accepts_plain_text() -> None:
    parsed = parse_grade_documents("yes")
    assert parsed.binary_score == "yes"


def test_parse_grade_documents_accepts_json() -> None:
    parsed = parse_grade_documents('{"binary_score": "no"}')
    assert parsed.binary_score == "no"


def test_parse_grade_documents_accepts_fenced_json() -> None:
    parsed = parse_grade_documents('```json\n{"binary_score": "yes"}\n```')
    assert parsed.binary_score == "yes"


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "What is the capital of France?"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"question": question, "context": docs}
    )
    pprint("--- GENERATION CHAIN OUTPUT ---")
    pprint(generation)


def test_hallucination_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"question": question, "context": docs}
    )
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    print("--- HALLUCINATION GRADER OUTPUT ---")
    pprint(res)
    assert res.binary_score

def test_hallucination_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": "pizzas are made of cheese and bread"}
    )
    print("--- HALLUCINATION GRADER OUTPUT ---")
    pprint(res)
    assert res.binary_score == "no"