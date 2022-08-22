from dataclasses import dataclass, field


@dataclass(order=True)
class BirdData:
    id: int = field(compare=False)
    name: str = field(compare=False)
    score: float = 0.0
