"""Entry point for minirepo."""

from .service import UserService


def main():
    service = UserService()
    service.create_user("alice")
    service.promote("bob")
    return service


def run():
    return main()
