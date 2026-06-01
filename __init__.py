from .pixai_tagger import PixAITagger, PixAITaggerFromPath

NODE_CLASS_MAPPINGS = {
    "PixAITagger": PixAITagger,
    "PixAITaggerFromPath": PixAITaggerFromPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixAITagger": "PixAI Tagger",
    "PixAITaggerFromPath": "PixAI Tagger From Path/URL",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
