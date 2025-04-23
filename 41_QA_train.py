from Question_answering.question_answering_service import QuestionAnsweringService


def main():
    question_answering_service = QuestionAnsweringService()
    n_epochs=1
    question_answering_service.handle_question_answering(n_epochs)

if __name__ == "__main__":
    main()
