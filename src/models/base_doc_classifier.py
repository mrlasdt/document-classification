from abc import ABC, abstractmethod


class BaseDocClasifier:
    @abstractmethod
    def read_input(input_path: str):
        pass

    @abstractmethod
    def classify(self, input_path: str, apply_ocr: bool) -> str:
        pass
