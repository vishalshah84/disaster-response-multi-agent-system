"""
Core utility functions for the simulation system
"""
import math
import numpy as np
from typing import Tuple, List, Optional


class Position:
    """Represents a 2D position in the grid"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def manhattan_distance_to(self, other: 'Position') -> int:
        """Calculate Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def is_adjacent(self, other: 'Position') -> bool:
        """Check if position is adjacent (including diagonals)"""
        return self.distance_to(other) <= math.sqrt(2) + 0.01


def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(pos: Tuple[int, int], width: int, height: int, 
                  diagonal: bool = True) -> List[Tuple[int, int]]:
    """Get valid neighbor positions"""
    x, y = pos
    neighbors = []
    
    # 4-directional movement
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Add diagonal if enabled
    if diagonal:
        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    
    return neighbors


def generate_random_position(width: int, height: int, 
                             exclude: Optional[List[Tuple[int, int]]] = None) -> Tuple[int, int]:
    """Generate a random position in the grid"""
    if exclude is None:
        exclude = []
    
    max_attempts = 100
    for _ in range(max_attempts):
        pos = (np.random.randint(0, width), np.random.randint(0, height))
        if pos not in exclude:
            return pos
    
    # Fallback: return any position
    return (np.random.randint(0, width), np.random.randint(0, height))


class IDGenerator:
    """Generate unique IDs for entities"""
    _counters = {}
    
    @classmethod
    def generate(cls, prefix: str) -> str:
        """Generate a unique ID with given prefix"""
        if prefix not in cls._counters:
            cls._counters[prefix] = 0
        cls._counters[prefix] += 1
        return f"{prefix}_{cls._counters[prefix]:04d}"
    
    @classmethod
    def reset(cls, prefix: Optional[str] = None):
        """Reset counter for a prefix or all prefixes"""
        if prefix:
            cls._counters[prefix] = 0
        else:
            cls._counters = {}
