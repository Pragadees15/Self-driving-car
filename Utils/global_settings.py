from Utils.misc_funcs import stdfrm


SAVED_MAPS_ROOT = r"saved_maps/"
SAVED_MODELS_ROOT = r"saved_models/"

# ======== WINDOW SETTINGS ========
FPS = 0  # Changed from 0 (unlimited) to 30 fps for stable visualization
GRID_SIZE_PIXELS = 60  # Increased from 30 to 60 for better resolution
HEIGHT = 1080  # Increased from 541 for higher resolution display
WIDTH = 1920  # Increased from 961 for higher resolution display
SF = GRID_SIZE_PIXELS/60  # scale factor


# ======== SOCKET SETTINGS ========
PORT = 5656
USE_UNREAL_SOCKET = False
MESSAGE_LENGTH = 60  # padded to this length with @


# ======== COLOURS ========
COL_BACKGROUND = (40, 60, 35)  # Darker background for better contrast
COL_GRID = (20, 30, 20)        # Darker grid lines for subtler appearance
COL_MOUSE_HIGHLIGHT = (109, 163, 77)
COL_PLACED_ROAD = (100, 100, 100)  # Slightly brighter roads


# ======== MISC ========
FREE_ROAM = False  # no collision


# ======== Deep Q Learning ========
LOAD_MODEL = "combined_model_02.10;22.45_0"
MAX_EPISODE_FRAMES = 4000

Q_LEARNING_SETTINGS = {
    "TRAINING": True,

    "MOVEMENT_PER_FRAME": 3,  # 1 is normal, 2 is double etc (can make predicting future reward easier due to less states)

    "LEARNING_RATE": stdfrm(1, -4),  # Increased learning rate for faster convergence

    "GD_MOMENTUM": 0.9,  # Added momentum for optimizer

    "DISCOUNT_RATE": 0.99,  # Increased to value future rewards more

    "EPSILON_PROBABILITY": 0.2,  # Slightly increased for more exploration
    "EPSILON_DECAY": 0.00005,  # Faster decay for quicker convergence
    "EPSILON_MIN": 0.01,  # Lower minimum for better final policy

    "TARGET_NET_COPY_STEPS": 2000,  # Reduced for faster target network updates
    "TRAIN_AMOUNT": 1.0,  # Train on all available data

    "BUFFER_LENGTH": 50000,  # Increased for more diverse experiences
    
    "BATCH_SIZE": 128,  # Batch size for GPU training (larger batches better utilize GPU)
    
    "USE_BATCH_NORM": True  # Whether to use batch normalization
}

# backward compatibility
TRAINING = Q_LEARNING_SETTINGS["TRAINING"]
