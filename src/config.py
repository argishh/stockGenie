from dataclasses import dataclass


@dataclass
class ModelConfig:
    INPUT_SIZE: int = 1
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 3
    OUTPUT_SIZE: int = 1
    DROPOUT: float = 0.2
    SEQUENCE_LENGTH: int = 60
    PREDICTION_DAYS: int = 30
    BATCH_SIZE: int = 32
    EPOCHS: int = 25
    TESTING_EPOCHS: int = 250
    LEARNING_RATE: float = 0.001

@dataclass
class AppConfig:
    # Base directories
    BASE_LOG_DIR: str = "./src/logs"
    MODEL_DIR: str = "./src/models"
    DATA_DIR: str = "./src/data"
    TEST_DIR: str = "./tests"
    
    # Log subdirectories
    LOG_DIRS: dict = None
    
    def __post_init__(self):
        self.LOG_DIRS = {
            "app": f"{self.BASE_LOG_DIR}/app",
            "model": f"{self.BASE_LOG_DIR}/model",
            "data": f"{self.BASE_LOG_DIR}/data",
            "metrics": f"{self.BASE_LOG_DIR}/metrics",
            "optimization": f"{self.BASE_LOG_DIR}/optimization",
            "test": {
                "unit": f"{self.BASE_LOG_DIR}/test/unit",
                "integration": f"{self.BASE_LOG_DIR}/test/integration",
                "performance": f"{self.BASE_LOG_DIR}/test/performance",
                "model": f"{self.BASE_LOG_DIR}/test/model",
            }
        }
    
    MAX_MODEL_VERSIONS: int = 3
    MODEL_FILENAME_TEMPLATE: str = "{ticker}_model_v{version}.pt"
