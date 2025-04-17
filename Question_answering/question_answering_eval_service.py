from pydantic import BaseModel
from transformers import pipeline

class QuestionAnsweringEvalService(BaseModel):
    def evaluate_question_answering(self, context, question):
        qa = pipeline('question-answering',
              model='Question_answering/saved_model',
              device=0)
        print('Answer to your question: ', qa(context=context, question=question), '\n')