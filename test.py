from dataclasses import InitVar, asdict, dataclass, field

class Interface:
    def f(self) -> None:
        print(113)

@dataclass(frozen=True)
class Test(Interface):
    a: int = 1
    b: float = field(default_factory=lambda: 0.0,)
    c: InitVar[int] = 1

    def __post_init__(self, c: int) -> None:
        object.__setattr__(self, "b", c * 2.0)



d = {"a": 2, "b": 0.1}
test = Test(**d)
print(test)
print(test.__dict__)
di = asdict(test)
di.pop("b")
print(di)
print(test.__dict__)
