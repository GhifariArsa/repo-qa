"""Model hierarchy: Base <- User <- Admin, and Base <- Guest."""

from .config import LOG_LEVEL


class Base:
    def __init__(self):
        self.created_at = None
        self.log_level = LOG_LEVEL

    def describe(self):
        return "base object"

    def validate(self):
        return self.created_at is not None


class User(Base):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.role = "user"

    def greet(self):
        return "hello " + self.name

    def validate(self):
        return bool(self.name)


class Admin(User):
    def __init__(self, name):
        super().__init__(name)
        self.permissions = []

    def grant(self, permission):
        self.permissions.append(permission)
        return self.describe()


class Guest(Base):
    def __init__(self):
        super().__init__()
        self.name = "guest"
