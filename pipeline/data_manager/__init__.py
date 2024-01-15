from pipeline.data_manager.llama_case_generation import CaseGenerationDataManager
from pipeline.data_manager.llama_feedback import FeedbackDataManager
from pipeline.data_manager.llama_sentence_analyzer import SentenceAnalyzerDataManager
from pipeline.data_manager.translation import TranslationDataManager
from pipeline.data_manager.alpaca_translation import AlpacaTranslationDataManager
from pipeline.data_manager.llama_word_analogy import WordAnalogyDataManager

def make_data_manager(args):
    if args.task == "feedback":
        return FeedbackDataManager()
    elif args.task == "sentence_analysis":
        return SentenceAnalyzerDataManager()
    elif args.task == "case_generation":
        return CaseGenerationDataManager()
    elif args.task == "translation":
        return TranslationDataManager()
    elif args.task == "translation_alpaca":
        return AlpacaTranslationDataManager()
    elif args.task == "word_analogy":
        return WordAnalogyDataManager()
    else:
        raise ValueError("Unknown task type")