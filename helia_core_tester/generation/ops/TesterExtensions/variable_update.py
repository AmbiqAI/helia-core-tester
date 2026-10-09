"""
VariableUpdate operation implementation.
"""

from helia_core_tester.generation.ops._shared.base import OperationBase


class OpVariableUpdate(OperationBase):
    """
    VariableUpdate operation - variable read/write operations.
    """
    
    def generate_c_files(self, output_dir) -> None:
        """
        VariableUpdate is not supported by CMSIS-NN.
        
        VariableUpdate is a stateful operation that involves reading and writing
        to persistent variables. CMSIS-NN is designed for stateless inference
        operations and does not provide kernels for variable management.
        
        Raises:
            NotImplementedError: Always, as VariableUpdate is not supported
        """
        raise NotImplementedError(
            "VariableUpdate is not supported by CMSIS-NN. "
            "VariableUpdate is a stateful operation that requires persistent "
            "variable storage, which is outside the scope of CMSIS-NN's stateless "
            "inference kernels. Consider using stateless operations or handling "
            "variable updates at a higher level in your application."
        )
