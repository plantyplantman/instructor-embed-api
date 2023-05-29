from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('instructor-large')

res = model.encode([["I am a student", "Represent the query for retrieval"]])
print(res)
