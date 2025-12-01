"""
Pathfinding algorithms - A* and related utilities
"""
import heapq
from typing import List, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field

from app.core.utils import manhattan_distance, get_neighbors


@dataclass(order=True)
class Node:
    """Node for A* algorithm"""
    f_score: float
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False)
    h_score: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)


def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Heuristic function for A* (Manhattan distance)"""
    return manhattan_distance(pos1, pos2)


def reconstruct_path(node: Node) -> List[Tuple[int, int]]:
    """Reconstruct path from goal node to start"""
    path = []
    current = node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return list(reversed(path))


def astar(start: Tuple[int, int], 
          goal: Tuple[int, int],
          width: int,
          height: int,
          is_walkable: Callable[[Tuple[int, int]], bool],
          diagonal: bool = True) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        width: Grid width
        height: Grid height
        is_walkable: Function that returns True if position is walkable
        diagonal: Allow diagonal movement
    
    Returns:
        List of positions from start to goal, or None if no path exists
    """
    # Early exit if start or goal is not walkable
    if not is_walkable(start) or not is_walkable(goal):
        return None
    
    # Initialize
    open_set = []
    closed_set: Set[Tuple[int, int]] = set()
    
    start_node = Node(
        f_score=heuristic(start, goal),
        position=start,
        g_score=0,
        h_score=heuristic(start, goal),
        parent=None
    )
    
    heapq.heappush(open_set, start_node)
    g_scores = {start: 0}
    
    while open_set:
        current = heapq.heappop(open_set)
        
        # Goal reached
        if current.position == goal:
            return reconstruct_path(current)
        
        # Skip if already processed
        if current.position in closed_set:
            continue
        
        closed_set.add(current.position)
        
        # Check neighbors
        for neighbor_pos in get_neighbors(current.position, width, height, diagonal):
            if neighbor_pos in closed_set:
                continue
            
            if not is_walkable(neighbor_pos):
                continue
            
            # Calculate tentative g_score
            # Diagonal movement costs sqrt(2), straight movement costs 1
            if diagonal:
                dx = abs(neighbor_pos[0] - current.position[0])
                dy = abs(neighbor_pos[1] - current.position[1])
                move_cost = 1.414 if (dx + dy) == 2 else 1.0
            else:
                move_cost = 1.0
            
            tentative_g = current.g_score + move_cost
            
            # Check if this path to neighbor is better
            if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                g_scores[neighbor_pos] = tentative_g
                h = heuristic(neighbor_pos, goal)
                f = tentative_g + h
                
                neighbor_node = Node(
                    f_score=f,
                    position=neighbor_pos,
                    g_score=tentative_g,
                    h_score=h,
                    parent=current
                )
                
                heapq.heappush(open_set, neighbor_node)
    
    # No path found
    return None


class PathPlanner:
    """
    Path planning utility with caching and replanning
    """
    
    def __init__(self, width: int, height: int, 
                 is_walkable: Callable[[Tuple[int, int]], bool]):
        self.width = width
        self.height = height
        self.is_walkable = is_walkable
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.current_goal: Optional[Tuple[int, int]] = None
        self.path_index = 0
    
    def plan_path(self, start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan a path from start to goal"""
        path = astar(start, goal, self.width, self.height, 
                    self.is_walkable, diagonal=True)
        
        self.current_path = path
        self.current_goal = goal
        self.path_index = 0
        
        return path
    
    def get_next_position(self) -> Optional[Tuple[int, int]]:
        """Get next position in current path"""
        if not self.current_path or self.path_index >= len(self.current_path):
            return None
        
        pos = self.current_path[self.path_index]
        self.path_index += 1
        return pos
    
    def needs_replanning(self, current_pos: Tuple[int, int]) -> bool:
        """Check if path needs replanning"""
        if not self.current_path:
            return True
        
        # Check if current position deviates from path
        if self.path_index < len(self.current_path):
            expected_pos = self.current_path[self.path_index]
            if manhattan_distance(current_pos, expected_pos) > 2:
                return True
        
        # Check if any position ahead in path is now blocked
        for i in range(self.path_index, min(self.path_index + 5, len(self.current_path))):
            if not self.is_walkable(self.current_path[i]):
                return True
        
        return False
    
    def replan(self, current_pos: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Replan path from current position to current goal"""
        if not self.current_goal:
            return None
        
        return self.plan_path(current_pos, self.current_goal)
    
    def clear(self):
        """Clear current path"""
        self.current_path = None
        self.current_goal = None
        self.path_index = 0
