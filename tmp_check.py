from graph.chains.retrieval_grader import parse_grade_documents

print(parse_grade_documents('yes').binary_score)
print(parse_grade_documents('{"binary_score": "yes"}').binary_score)
