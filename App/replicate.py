"""
Simplified version of ReplicatedTransform that does not rely on SocketCommunication
"""

class ReplicatedTransform:
    """
    Simple base class for objects that need to be tracked with a transform
    """
    def __init__(self, object_type, object_name, target_type="loc"):
        self.object_type = object_type
        self.object_name = object_name
        self.target_type = target_type
        
    def get_data_to_replicate(self):
        """Method to be overridden by child classes"""
        return None 