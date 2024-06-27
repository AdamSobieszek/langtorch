from queue import Queue
from threading import Lock, Thread
from typing import List, Tuple, Sequence

import torch

from langtorch.torch_utils import _OptionalTensor


def grad_worker_func(queue, gradients, retain_graph, create_graph, lock):
    """
    The function that each worker thread runs in the parallel backward.

    Args:
        queue: The queue of nodes to process in the computational graph.
        gradients: The map to store computed gradients.
        retain_graph: Whether to retain the computational graph for further backward passes.
        create_graph: Whether to create a computational graph for higher order gradients.
        lock: A threading lock for thread-safety.
    """
    from langtorch import TextTensor

    # While there are nodes in the queue
    while not queue.empty():
        # Get a node from the queue
        with lock:
            node, grad_tensor = queue.get()

        # If the node is not a leaf node
        if node is not None:
            save = False
            # Execute backward
            if hasattr(node, "apply"):
                # print(node)
                # for x in node.apply(grad_tensor):
                #     if isinstance(x, torch.Tensor):
                #         print(x.content)
                grads = node.apply(grad_tensor)
                for x in grads:
                    if isinstance(x, TextTensor) and not hasattr(x, "backward_activation"):
                        x.backward_activation = grad_tensor.backward_activation
                # if not hasattr(grads, "backward_activation"):
                #     grads.backward_activation = grad_tensor.backward_activation
            elif hasattr(node, "__call__"):
                if type(node).__name__ == "AccumulateGrad":
                    save = True
                    grads = grad_tensor
            else:
                print("An unsupported backward graph node:", node, "dir: ", dir(node))

            # Store the computed gradients
            if save:
                with lock:
                    gradients.append((node.variable, grads))

            # If there are parent nodes to backpropagate through
            if node.next_functions:
                # Put the parent nodes in the queue
                for (next_node, _), grad in zip(node.next_functions, grads):
                    with lock:
                        queue.put((next_node, grad))


def run_backward(tensors, grad_tensors_, retain_graph, create_graph, inputs, allow_unreachable=True,
                 accumulate_grad=True):
    """
    Run the backward pass using the execution engine.

    Args:
        tensors: List of tensors for which to compute gradients.
        grad_tensors_: Gradients of the output with respect to the tensors.
        retain_graph: Whether to retain the computational graph for further backward passes.
        create_graph: Whether to create a computational graph for higher order gradients.
        inputs: Input nodes to start backpropagation from.
        allow_unreachable: Whether to allow unreachable nodes. If False, an error will be raised if not all nodes can be reached.
        accumulate_grad: Whether to accumulate gradients in `.grad` variables.
    """

    # 1. Check if any of the tensors require gradients
    for tensor in tensors:
        assert tensor.requires_grad, f"Tensor {tensor} does not require gradients.\n{tensors}"

    # 2. Prepare an empty list to store computed gradients
    gradients = []

    # 3. Prepare a queue for nodes to be processed in the graph and lock for thread-safety
    queue = Queue()
    lock = Lock()

    # 4. Initialize queue with starting nodes
    for tensor, grad in zip(tensors, grad_tensors_):
        queue.put((tensor.grad_fn, grad))
    # PARALLELIZED PART STARTS HERE
    # 5. Spawn worker threads for backpropagation
    worker_threads = []
    NUM_WORKERS = 8
    for _ in range(NUM_WORKERS):
        worker_thread = Thread(target=grad_worker_func, args=(queue, gradients, retain_graph, create_graph, lock))
        worker_thread.start()
        worker_threads.append(worker_thread)

    # 6. Wait for all worker threads to finish
    for worker_thread in worker_threads:
        worker_thread.join()
    # PARALLELIZED PART ENDS HERE

    # 9. If we are accumulating gradients, add the computed gradients to the `.grad` attributes of the tensors
    for tensor, grad in gradients:
        # print("gradient assignment: ", tensor)
        if tensor.requires_grad and grad is not None:
            # print(tensor, type(grad))
            if (tensor.is_leaf or tensor.retain_grad):
                if accumulate_grad and tensor.grad is not None:
                    tensor.grad = tensor.grad + grad.reshape(tensor.shape)

                elif tensor.grad is None or not accumulate_grad:
                    tensor.grad = grad.reshape(tensor.shape)

    return


def _calculate_shape(output: torch.Tensor, grad: torch.Tensor,
                     is_grads_batched: bool):
    # is_same_size ensures that both tensors are either nested or non nested
    if output.is_nested:
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with Nested Tensor.")
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()

        return out_shape, grad_shape

    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return reg_out_shape, reg_grad_shape


def make_grads(outputs: Sequence[torch.Tensor], grads: Sequence[_OptionalTensor],
               is_grads_batched: bool) -> Tuple[_OptionalTensor, ...]:
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, torch.Tensor):
            first_grad = grad if not is_grads_batched else grad[0]
            if not torch.is_same_size(out, first_grad):
                out_shape, grad_shape = _calculate_shape(out, first_grad, is_grads_batched)
                if is_grads_batched:
                    raise RuntimeError("If `is_grads_batched=True`, we interpret the first "
                                       "dimension of each grad_output as the batch dimension. "
                                       "The sizes of the remaining dimensions are expected to match "
                                       "the shape of corresponding output, but a mismatch "
                                       "was detected: grad_output["
                                       + str(grads.index(grad)) + "] has a shape of "
                                       + str(grad_shape) + " and output["
                                       + str(outputs.index(out)) + "] has a shape of "
                                       + str(out_shape) + ". "
                                                          "If you only want some tensors in `grad_output` to be considered "
                                                          "batched, consider using vmap.")
                else:
                    raise RuntimeError("Mismatch in shape: grad_output["
                                       + str(grads.index(grad)) + "] has a shape of "
                                       + str(grad_shape) + " and output["
                                       + str(outputs.index(out)) + "] has a shape of "
                                       + str(out_shape) + ".")
            if out.dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError("For complex Tensors, both grad_output and output"
                                   " are required to have the same dtype."
                                   " Mismatch in dtype: grad_output["
                                   + str(grads.index(grad)) + "] has a dtype of "
                                   + str(grad.dtype) + " and output["
                                   + str(outputs.index(out)) + "] has a dtype of "
                                   + str(out.dtype) + ".")
            new_grads.append(grad)
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads)
