from Question_answering.question_answering_eval_service import QuestionAnsweringEvalService


def main():
    question_answering_eval_service = QuestionAnsweringEvalService()
    context = 'I went to the store to build a carton of milk.'
    question = 'What did I buy?'
    question_answering_eval_service.evaluate_question_answering(context, question)

if __name__ == "__main__":
    main()
