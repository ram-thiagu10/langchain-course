from dotenv import load_dotenv

load_dotenv()
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever

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