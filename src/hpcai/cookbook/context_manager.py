from hpcai import TrainingClient

class UnloadContextManager:
    def __init__(self, training_client: TrainingClient | None = None):
        self.training_client = training_client
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc, tb):
        print("Exiting: Unloading training client resources...")
        self.training_client.unload_model().result()
        return False  # Do not suppress exceptions