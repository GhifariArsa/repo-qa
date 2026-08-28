"""Service layer wiring models, config and utils together."""

from .config import MAX_RETRIES, get_timeout
from .models import Admin, User
from .utils import normalize


class UserService:
    def __init__(self):
        self.users = []
        self.timeout = get_timeout()

    def create_user(self, name):
        clean = normalize(name)
        user = User(clean)
        self.users.append(user)
        return user

    def promote(self, name):
        admin = Admin(normalize(name))
        return admin

    def retry_count(self):
        return MAX_RETRIES
