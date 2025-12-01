"""
Environment module - Grid world for disaster simulation
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from app.core.config import DisasterType, settings
from app.core.utils import Position, generate_random_position, get_neighbors


class CellType(Enum):
    """Types of cells in the grid"""
    EMPTY = 0
    OBSTACLE = 1
    FIRE = 2
    FLOOD = 3
    BUILDING = 4
    RESOURCE_POINT = 5
    VICTIM = 6


@dataclass
class Disaster:
    """Represents a disaster in the environment"""
    id: str
    type: str
    position: Position
    intensity: float  # 0.0 to 1.0
    spread_rate: float
    active: bool = True
    victims_count: int = 0
    
    def update(self, delta_time: float) -> bool:
        """Update disaster state, returns True if still active"""
        if not self.active:
            return False
        
        # Decrease intensity over time
        self.intensity = max(0, self.intensity - 0.001 * delta_time)
        
        if self.intensity <= 0:
            self.active = False
            return False
        
        return True


@dataclass
class Resource:
    """Represents a resource point in the environment"""
    id: str
    position: Position
    type: str  # "medical", "fuel", "food", "water"
    amount: int
    max_amount: int
    
    def consume(self, amount: int) -> int:
        """Consume resource, returns actual amount consumed"""
        actual = min(amount, self.amount)
        self.amount -= actual
        return actual
    
    def is_depleted(self) -> bool:
        """Check if resource is depleted"""
        return self.amount <= 0


@dataclass
class Victim:
    """Represents a victim needing rescue"""
    id: str
    position: Position
    health: float  # 0.0 to 1.0
    rescued: bool = False
    
    def update(self, delta_time: float):
        """Update victim state (health deteriorates over time)"""
        if not self.rescued:
            self.health = max(0, self.health - 0.01 * delta_time)


class Environment:
    """Main environment class for the disaster simulation"""
    
    def __init__(self, width: int = None, height: int = None):
        self.width = width or settings.GRID_WIDTH
        self.height = height or settings.GRID_HEIGHT
        
        # Grid representation
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # Entities
        self.disasters: Dict[str, Disaster] = {}
        self.resources: Dict[str, Resource] = {}
        self.victims: Dict[str, Victim] = {}
        self.obstacles: Set[Tuple[int, int]] = set()
        self.buildings: Set[Tuple[int, int]] = set()
        
        # Simulation state
        self.time_elapsed = 0.0
        self.disasters_spawned = 0
        self.victims_saved = 0
        self.victims_lost = 0
        
        # Initialize environment
        self._initialize()
    
    def _initialize(self):
        """Initialize the environment with obstacles and buildings"""
        # Add some random obstacles
        num_obstacles = int(self.width * self.height * 0.05)
        for _ in range(num_obstacles):
            pos = generate_random_position(self.width, self.height)
            self.add_obstacle(pos)
        
        # Add buildings
        num_buildings = int(self.width * self.height * 0.1)
        for _ in range(num_buildings):
            pos = generate_random_position(self.width, self.height, 
                                          exclude=list(self.obstacles))
            self.add_building(pos)
        
        # Add initial resource points
        resource_types = ["medical", "fuel", "food", "water"]
        for i, res_type in enumerate(resource_types):
            for j in range(3):
                pos = generate_random_position(
                    self.width, self.height,
                    exclude=list(self.obstacles) + list(self.buildings)
                )
                self.add_resource(
                    f"resource_{res_type}_{j}",
                    pos,
                    res_type,
                    amount=100
                )
    
    def add_obstacle(self, position: Tuple[int, int]):
        """Add an obstacle to the environment"""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add(position)
            self.grid[y, x] = CellType.OBSTACLE.value
    
    def add_building(self, position: Tuple[int, int]):
        """Add a building to the environment"""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.buildings.add(position)
            self.grid[y, x] = CellType.BUILDING.value
    
    def spawn_disaster(self, disaster_type: str, 
                       position: Optional[Position] = None,
                       intensity: float = 0.8) -> Disaster:
        """Spawn a new disaster"""
        if position is None:
            pos_tuple = generate_random_position(
                self.width, self.height,
                exclude=list(self.obstacles)
            )
            position = Position(pos_tuple[0], pos_tuple[1])
        
        disaster_id = f"disaster_{self.disasters_spawned}"
        self.disasters_spawned += 1
        
        # Determine spread rate based on type
        spread_rates = {
            DisasterType.FIRE: 0.3,
            DisasterType.FLOOD: 0.2,
            DisasterType.EARTHQUAKE: 0.1,
        }
        
        disaster = Disaster(
            id=disaster_id,
            type=disaster_type,
            position=position,
            intensity=intensity,
            spread_rate=spread_rates.get(disaster_type, 0.1),
            victims_count=np.random.randint(1, 5)
        )
        
        self.disasters[disaster_id] = disaster
        
        # Update grid
        x, y = position.x, position.y
        if disaster_type == DisasterType.FIRE:
            self.grid[y, x] = CellType.FIRE.value
        elif disaster_type == DisasterType.FLOOD:
            self.grid[y, x] = CellType.FLOOD.value
        
        # Spawn victims at disaster location
        for i in range(disaster.victims_count):
            victim = Victim(
                id=f"victim_{disaster_id}_{i}",
                position=position,
                health=1.0
            )
            self.victims[victim.id] = victim
        
        return disaster
    
    def add_resource(self, res_id: str, position, 
                     res_type: str, amount: int):
        """Add a resource point"""
        # Convert tuple to Position if needed
        if isinstance(position, tuple):
            position = Position(position[0], position[1])
        
        resource = Resource(
            id=res_id,
            position=position,
            type=res_type,
            amount=amount,
            max_amount=amount
        )
        self.resources[res_id] = resource
        
        x, y = position.x, position.y
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = CellType.RESOURCE_POINT.value
    
    def resolve_disaster(self, disaster_id: str) -> bool:
        """Mark a disaster as resolved"""
        if disaster_id in self.disasters:
            disaster = self.disasters[disaster_id]
            disaster.active = False
            disaster.intensity = 0
            
            x, y = disaster.position.x, disaster.position.y
            self.grid[y, x] = CellType.EMPTY.value
            
            return True
        return False
    
    def rescue_victim(self, victim_id: str) -> bool:
        """Mark a victim as rescued"""
        if victim_id in self.victims:
            victim = self.victims[victim_id]
            if not victim.rescued and victim.health > 0:
                victim.rescued = True
                self.victims_saved += 1
                return True
        return False
    
    def is_walkable(self, position: Tuple[int, int]) -> bool:
        """Check if a position is walkable"""
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        return (x, y) not in self.obstacles
    
    def get_disasters_in_radius(self, position: Position, 
                                radius: float) -> List[Disaster]:
        """Get all disasters within a radius of a position"""
        disasters = []
        for disaster in self.disasters.values():
            if disaster.active and disaster.position.distance_to(position) <= radius:
                disasters.append(disaster)
        return disasters
    
    def get_resources_in_radius(self, position: Position, 
                                radius: float) -> List[Resource]:
        """Get all resources within a radius of a position"""
        resources = []
        for resource in self.resources.values():
            if not resource.is_depleted() and resource.position.distance_to(position) <= radius:
                resources.append(resource)
        return resources
    
    def get_victims_in_radius(self, position: Position, 
                             radius: float) -> List[Victim]:
        """Get all unrescued victims within a radius"""
        victims = []
        for victim in self.victims.values():
            if not victim.rescued and victim.health > 0:
                if victim.position.distance_to(position) <= radius:
                    victims.append(victim)
        return victims
    
    def update(self, delta_time: float):
        """Update environment state"""
        self.time_elapsed += delta_time
        
        # Update disasters
        disasters_to_remove = []
        for disaster_id, disaster in self.disasters.items():
            if not disaster.update(delta_time):
                disasters_to_remove.append(disaster_id)
                x, y = disaster.position.x, disaster.position.y
                if self.grid[y, x] in [CellType.FIRE.value, CellType.FLOOD.value]:
                    self.grid[y, x] = CellType.EMPTY.value
        
        for disaster_id in disasters_to_remove:
            del self.disasters[disaster_id]
        
        # Update victims
        for victim in self.victims.values():
            victim.update(delta_time)
            if victim.health <= 0 and not victim.rescued:
                self.victims_lost += 1
    
    def get_state_dict(self) -> Dict:
        """Get current environment state as dictionary"""
        return {
            "time_elapsed": self.time_elapsed,
            "grid": self.grid.tolist(),
            "disasters": [
                {
                    "id": d.id,
                    "type": d.type,
                    "position": {"x": d.position.x, "y": d.position.y},
                    "intensity": d.intensity,
                    "active": d.active,
                    "victims_count": d.victims_count
                }
                for d in self.disasters.values()
            ],
            "resources": [
                {
                    "id": r.id,
                    "type": r.type,
                    "position": {"x": r.position.x, "y": r.position.y},
                    "amount": r.amount,
                    "max_amount": r.max_amount
                }
                for r in self.resources.values()
            ],
            "victims": [
                {
                    "id": v.id,
                    "position": {"x": v.position.x, "y": v.position.y},
                    "health": v.health,
                    "rescued": v.rescued
                }
                for v in self.victims.values()
            ],
            "stats": {
                "victims_saved": self.victims_saved,
                "victims_lost": self.victims_lost,
                "active_disasters": len([d for d in self.disasters.values() if d.active])
            }
        }
    
    def reset(self):
        """Reset environment to initial state"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.disasters.clear()
        self.resources.clear()
        self.victims.clear()
        self.obstacles.clear()
        self.buildings.clear()
        self.time_elapsed = 0.0
        self.disasters_spawned = 0
        self.victims_saved = 0
        self.victims_lost = 0
        self._initialize()
