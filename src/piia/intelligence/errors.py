class PiiaError(Exception):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code

class NotFound(PiiaError):
    def __init__(self, message: str):
        super().__init__(message, "not_found")

class Conflict(PiiaError):
    def __init__(self, message: str):
        super().__init__(message, "conflict")

class Immutable(PiiaError):
    def __init__(self, message: str = "original records cannot be updated or deleted"):
        super().__init__(message, "immutable")

class ValidationFailed(PiiaError):
    def __init__(self, message: str):
        super().__init__(message, "validation")
