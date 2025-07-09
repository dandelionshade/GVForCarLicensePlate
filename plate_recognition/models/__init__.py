# -*- coding: utf-8 -*-
"""
Models module - Deep learning models for license plate recognition
"""

# Lazy imports to avoid heavy dependencies at package level
def get_crnn_model():
    """Get CRNN model instance (lazy loading)"""
    from .crnn_model import CRNNModel
    return CRNNModel()

# Direct imports for convenience
try:
    from .crnn_model import CRNNModel
    __all__ = ['CRNNModel', 'get_crnn_model']
except ImportError:
    # TensorFlow not available
    __all__ = ['get_crnn_model']
