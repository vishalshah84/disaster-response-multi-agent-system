# Core configuration
class Settings:
    APP_NAME = "Disaster Response Simulation System"
    APP_VERSION = "1.0.0"
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 8000
    GRID_WIDTH = 50
    GRID_HEIGHT = 50
    MAX_AGENTS = 50
    SIMULATION_SPEED = 0.3
    AGENT_VISION_RADIUS = 5
    AGENT_MAX_FUEL = 1000
    AGENT_MAX_SUPPLIES = 50
    AGENT_SPEED = 1.0
    FIRE_SPREAD_RATE = 0.1
    DISASTER_SPAWN_INTERVAL = 5
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.95
    EPSILON = 0.1

settings = Settings()

class AgentType:
    FIRE_FIGHTER = "fire_fighter"
    MEDICAL = "medical"
    SUPPLY = "supply"
    SEARCH_RESCUE = "search_rescue"

class BehaviorMode:
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    AGREEMENT = "agreement"

class DisasterType:
    FIRE = "fire"
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"

REWARDS = {
    "disaster_resolved": 100,
    "life_saved": 200,
    "resource_delivered": 50,
    "movement_penalty": -1,
}
