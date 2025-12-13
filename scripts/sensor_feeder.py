import json
import time
import random
import os
import sys

# --- ROBUST PATH SETUP ---
# 1. Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get project root
project_root = os.path.dirname(script_dir)
# 3. Define file paths
CONFIG_FILE = os.path.join(project_root, "data", "config.json")
SENSOR_FILE = os.path.join(project_root, "data", "live_sensors.json")

class SensorSimulation:
    def __init__(self):
        self.rows = 10
        self.cols = 10
        self.active_hazards = {} # Dictionary to track persistent hazards
        self.load_config()

    def load_config(self):
        """Reads the C++ simulation config to determine grid boundaries."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                # Support both structure formats seen in your project
                if "grid_size" in data:
                    self.rows = data["grid_size"].get("rows", 10)
                    self.cols = data["grid_size"].get("cols", 10)
                else:
                    self.rows = data.get("rows", 10)
                    self.cols = data.get("cols", 10)
            print(f"[Init] Loaded Grid Configuration: {self.rows}x{self.cols}")
        except FileNotFoundError:
            print(f"[Warn] Config file not found at {CONFIG_FILE}. Using default 10x10.")

    def update_hazards(self, step):
        """Updates the state of the world: spawns new fires, spreads existing ones."""
        
        # 1. Chance to spawn a NEW Fire (e.g., 5% chance per step)
        if random.random() < 0.05:
            r = random.randint(1, self.rows - 2) # Avoid edges/exits mostly
            c = random.randint(1, self.cols - 2)
            key = f"{r},{c}"
            if key not in self.active_hazards:
                self.active_hazards[key] = {
                    "id": f"THERMAL_{r}_{c}",
                    "type": "THERMAL",
                    "row": r, "col": c,
                    "value": 450.0,
                    "data": "FIRE_DETECTED",
                    "age": 0
                }
                print(f"[Event] FIRE started at ({r}, {c})")

        # 2. Chance to spawn moving SMOKE (e.g., 10% chance)
        if random.random() < 0.1:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            key = f"S_{r},{c}"
            self.active_hazards[key] = {
                "id": f"SMOKE_{r}_{c}",
                "type": "SMOKE",
                "row": r, "col": c,
                "value": 80.0,
                "data": "heavy",
                "age": 0,
                "is_moving": True
            }

        # 3. Process Updates (Spread Fire, Move Smoke)
        keys_to_remove = []
        new_hazards = {}

        for key, hazard in self.active_hazards.items():
            hazard["age"] += 1
            
            # Fire Spreading Logic: Every 5 steps, fire spreads to a neighbor
            if hazard["type"] == "THERMAL" and hazard["age"] % 5 == 0:
                # Pick a random neighbor
                nr = hazard["row"] + random.choice([-1, 0, 1])
                nc = hazard["col"] + random.choice([-1, 0, 1])
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    n_key = f"{nr},{nc}"
                    if n_key not in self.active_hazards and n_key not in new_hazards:
                        new_hazards[n_key] = {
                            "id": f"THERMAL_{nr}_{nc}",
                            "type": "THERMAL",
                            "row": nr, "col": nc,
                            "value": 400.0,
                            "data": "FIRE_DETECTED",
                            "age": 0
                        }
                        print(f"[Event] FIRE spread to ({nr}, {nc})")

            # Smoke Moving Logic: Smoke drifts right/down
            if hazard.get("is_moving") and hazard["age"] % 2 == 0:
                keys_to_remove.append(key) # Remove old position
                # Calculate new position (Drift Right)
                nc = hazard["col"] + 1
                if nc < self.cols:
                    new_key = f"S_{hazard['row']},{nc}"
                    hazard["col"] = nc
                    hazard["id"] = f"SMOKE_{hazard['row']}_{nc}"
                    new_hazards[new_key] = hazard

            # Remove old hazards (e.g., smoke dissipates after 15 steps)
            if hazard["type"] == "SMOKE" and hazard["age"] > 15:
                keys_to_remove.append(key)

        # Apply updates
        for k in keys_to_remove:
            if k in self.active_hazards:
                del self.active_hazards[k]
        self.active_hazards.update(new_hazards)

    def generate_readings(self):
        """Converts active hazards into the list format expected by C++."""
        readings = []
        current_time = time.time()
        for item in self.active_hazards.values():
            readings.append({
                "id": item["id"],
                "type": item["type"],
                "row": item["row"],
                "col": item["col"],
                "value": item["value"],
                "data": item["data"],
                "timestamp": current_time
            })
        return readings

def main():
    print("--- Random Dynamic Sensor Simulator ---")
    sim = SensorSimulation()
    
    step = 0
    try:
        while True:
            sim.update_hazards(step)
            data = sim.generate_readings()
            
            # Atomic write to ensure C++ doesn't read partial file
            temp_file = SENSOR_FILE + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=4)
            os.replace(temp_file, SENSOR_FILE)
            
            print(f"Step {step}: Active Hazards: {len(data)}")
            step += 1
            time.sleep(1.0) # Update rate (1 Hz)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

if __name__ == "__main__":
    main()