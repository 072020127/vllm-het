import importlib
import functools
import logging
import torch
import torch.distributed
from typing import Any, Callable, Optional, Union
from torch.distributed import Backend, ProcessGroup, ReduceOp
from types import ModuleType
import os, sys, time, traceback, importlib.util, importlib.abc

logger = logging.getLogger("sitecustomize")
logger.info("sitecustomize loaded, preparing vllm patch")

LOG_PATH = "/tmp/vllm_auto_patch.log"

TARGET = "vllm.distributed.parallel_state"

P2P_RANK_IP_CACHE = {}

_EXTRA_PATCHES = {
    "vllm.distributed.parallel_state": "patch_parallel_state",
    "vllm.utils": "patch_vllm_utils",
    "vllm.distributed.device_communicators.pynccl": "patch_pynccl_init",
}

def log(msg):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def load_p2p_backend():
    base_dir = os.path.dirname(__file__)
    p2p_path = os.path.join(base_dir, "p2p_backend.py")
    if not os.path.exists(p2p_path):
        return None
    # choose a unique module name to avoid colliding with other imports
    module_name = "vllm_auto_patch__p2p_backend"
    try:
        spec = importlib.util.spec_from_file_location(module_name, p2p_path)
        module = importlib.util.module_from_spec(spec)
        # optionally register in sys.modules so further imports by name find it
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        print("\n\n\nimport error\n\n\n")
        with open("/tmp/vllm_auto_patch.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} load_p2p_backend_from_file failed:\n")
            import traceback
            f.write(traceback.format_exc() + "\n")
        # make sure no half-loaded module remains
        sys.modules.pop(module_name, None)
        return None

p2p = load_p2p_backend()
if p2p == None:
    print("\n\n\nimport error\n\n\n")


def patch_parallel_state(mod: ModuleType):
    try:
        GC = getattr(mod, "GroupCoordinator", None)
        if GC is None:
            log("parallel_state.GroupCoordinator not found")
            return

        orig_init = getattr(GC, "__init__", None)
        orig_send_tensor_dict = getattr(GC, "send_tensor_dict", None)
        orig_recv_tensor_dict = getattr(GC, "recv_tensor_dict", None)

        def vllm_het_init(
            self,
            group_ranks: list[list[int]],
            local_rank: int,
            torch_distributed_backend: Union[str, Backend],
            use_device_communicator: bool,
            use_message_queue_broadcaster: bool = False,
            group_name: Optional[str] = None,
        ):
            orig_init(
                self,
                group_ranks,
                local_rank,
                torch_distributed_backend,
                use_device_communicator,
                use_message_queue_broadcaster,
                group_name,
            )
            self.use_hmc = True

            if group_name == "world":
                P2P_RANK_IP_CACHE["world"] = p2p.get_rank_ip_port_map(group = self.cpu_group)
            elif group_name == "pp":
                rank_ip = P2P_RANK_IP_CACHE.get("world", None)
                # p2p.init_p2p_comm(group = self.cpu_group, rankip = rank_ip)
                try:
                    device_id = int(local_rank)
                except Exception:
                    device_id = 0
                p2p.init_p2p_comm(group = self.cpu_group, rankip = rank_ip, device_id = device_id)

        def vllm_het_send(
            self,
            tensor_dict: dict[str, Union[torch.Tensor, Any]],
            dst: Optional[int] = None,
            all_gather_group: Optional["GroupCoordinator"] = None,
        ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
            """Send the input tensor dictionary.
            NOTE: `dst` is the local rank of the source rank.
            """

            # Bypass the function if we are using only 1 GPU.
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return tensor_dict

            all_gather_size = (1 if all_gather_group is None else
                            all_gather_group.world_size)
            all_gather_rank = (0 if all_gather_group is None else
                            all_gather_group.rank_in_group)

            group = self.device_group
            metadata_group = self.cpu_group

            if dst is None:
                dst = (self.rank_in_group + 1) % self.world_size
            assert dst < self.world_size, f"Invalid dst rank ({dst})"

            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict,
                dict), f"Expecting a dictionary, got {type(tensor_dict)}"
            metadata_list, tensor_list = mod._split_tensor_dict(tensor_dict)
            self.send_object(metadata_list, dst=dst)
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip sending empty tensors.
                    continue

                # send-allgather: send only a slice, then do allgather.
                if (all_gather_group is not None
                        and tensor.numel() % all_gather_size == 0):
                    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

                if self.use_hmc:
                    if tensor.is_cpu:
                        p2p._HET_COMM[metadata_group.group_name].send(tensor, dst=self.ranks[dst])
                    else:
                        p2p._HET_COMM[metadata_group.group_name].send(tensor, dst=self.ranks[dst])
                else:
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        torch.distributed.send(tensor,
                                            dst=self.ranks[dst],
                                            group=metadata_group)
                    else:
                        # use group for GPU tensors
                        torch.distributed.send(tensor,
                                            dst=self.ranks[dst],
                                            group=group)
            return None
        
        def vllm_het_recv(
            self,
            src: Optional[int] = None,
            all_gather_group: Optional["GroupCoordinator"] = None,
        ) -> Optional[dict[str, Union[torch.Tensor, Any]]]:
            
            if not torch.distributed.is_initialized() or self.world_size == 1:
                return None

            all_gather_size = (1 if all_gather_group is None else
                            all_gather_group.world_size)
            all_gather_rank = (0 if all_gather_group is None else
                            all_gather_group.rank_in_group)

            group = self.device_group
            metadata_group = self.cpu_group

            print(metadata_group.group_name)

            if src is None:
                src = (self.rank_in_group - 1) % self.world_size
            assert src < self.world_size, f"Invalid src rank ({src})"

            recv_metadata_list = self.recv_object(src=src)
            tensor_dict: dict[str, Any] = {}
            for key, value in recv_metadata_list:
                if isinstance(value, mod.TensorMetadata):
                    tensor = torch.empty(value.size,
                                        dtype=value.dtype,
                                        device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue

                    # send-allgather: send only a slice, then do allgather.
                    use_all_gather = (all_gather_group is not None
                                    and tensor.numel() % all_gather_size == 0)

                    if use_all_gather:
                        orig_shape = tensor.shape
                        tensor = tensor.reshape(all_gather_size,
                                                -1)[all_gather_rank]
                    
                    if self.use_hmc:
                        if tensor.is_cpu:
                            p2p._HET_COMM[metadata_group.group_name].recv(tensor, src=self.ranks[src])
                        else:
                            p2p._HET_COMM[metadata_group.group_name].recv(tensor, src=self.ranks[src])
                    else:
                        if tensor.is_cpu:
                            # use metadata_group for CPU tensors
                            torch.distributed.recv(tensor,
                                                src=self.ranks[src],
                                                group=metadata_group)
                        else:
                            # use group for GPU tensors
                            torch.distributed.recv(tensor,
                                                src=self.ranks[src],
                                                group=group)
                    if use_all_gather:
                        # do the allgather
                        tensor = all_gather_group.all_gather(  # type: ignore
                            tensor, dim=0)
                        tensor = tensor.reshape(orig_shape)

                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            return tensor_dict
        
        
        GC.__init__ = vllm_het_init
        GC.send_tensor_dict = vllm_het_send
        GC.recv_tensor_dict = vllm_het_recv

        log("patch applied OK: " + str(getattr(mod, "__file__", None)))
    except Exception:
        log("apply_patch failed:\n" + traceback.format_exc())

def patch_vllm_utils(mod: ModuleType):
    """
    Wrap vllm.utils.update_environment_variables(envs)
    - Default: behave like original, but log each change.
    - If env VLLM_PATCH_ENV_SKIP=1: skip overwriting existing env vars (do not change os.environ if key exists).
    - If env VLLM_PATCH_ENV_FORCE=1: force overwrite (same as original).
    """
    try:
        orig = getattr(mod, "update_environment_variables", None)
        if orig is None or not callable(orig):
            log("vllm.utils.update_environment_variables not found; skipping patch")
            return

        def wrapped_update_environment_variables(envs: dict[str, str]):
            for k, v in envs.items():
                # Skip overwriting 'LD_LIBRARY_PATH'
                if k == 'LD_LIBRARY_PATH':
                    if k in os.environ and os.environ[k] != v:
                        logger.warning(
                            "Attempt to overwrite environment variable %s from '%s' to '%s' was skipped",
                            k, os.environ[k], v
                        )
                    continue  # Skip the update if LD_LIBRARY_PATH is being changed
                
                # If the variable is different, log and update it
                if k in os.environ and os.environ[k] != v:
                    logger.warning(
                        "Overwriting environment variable %s from '%s' to '%s'",
                        k, os.environ[k], v
                    )
                
                os.environ[k] = v
            log("utils_init with new patch")

        setattr(mod, "update_environment_variables", wrapped_update_environment_variables)
        log("patch_vllm_utils applied")
    except Exception:
        log("patch_vllm_utils failed:\n" + traceback.format_exc())

import torch.distributed as dist

def patch_pynccl_init(mod: ModuleType):
    """
    Wrap PyNcclCommunicator.__init__ to optionally disable its initialization.
    - If env VLLM_PATCH_DISABLE_PYNCCL=1, skip calling original __init__ and mark instance._pynccl_disabled = True.
    - Otherwise call original __init__ and if it raises, catch, log, mark instance as disabled to avoid crash propagation.
    The wrapper uses *args/**kwargs so it is robust to signature changes.
    """
    try:
        PyNcclCommunicator = getattr(mod, "PyNcclCommunicator", None)
        if PyNcclCommunicator is None:
            log("pynccl.PyNcclCommunicator not found; skipping patch")
            return

        orig_init = getattr(PyNcclCommunicator, "__init__", None)
        if orig_init is None:
            log("PyNcclCommunicator.__init__ not found; skipping patch")
            return

        def wrapped_init(self, *args, **kwargs):
            logger.info("user new PyNcclCommunicator init")

        # replace the __init__ on the class
        wrapped_init.__wrapped__ = orig_init
        setattr(PyNcclCommunicator, "__init__", wrapped_init)
        log("patch_pynccl_init applied")
    except Exception:
        log("patch_pynccl_init failed:\n" + traceback.format_exc())


def _dispatch_patch_for_module(module: ModuleType):
    name = getattr(module, "__name__", None)
    if not name:
        return
    fn_name = _EXTRA_PATCHES.get(name)
    if not fn_name:
        return
    fn = globals().get(fn_name)
    if not fn:
        log(f"Patch function {fn_name} for module {name} not found")
        return
    try:
        log(f"Dispatching patch {fn_name} for module {name}")
        fn(module)
    except Exception:
        log(f"Dispatch of {fn_name} for module {name} failed:\n" + traceback.format_exc())


# If any target modules are already loaded, apply their patches immediately.
for _mod_name in list(_EXTRA_PATCHES.keys()):
    if _mod_name in sys.modules:
        try:
            _dispatch_patch_for_module(sys.modules[_mod_name])
        except Exception:
            log(f"initial dispatch for {_mod_name} failed:\n" + traceback.format_exc())


# Meta-path finder that intercepts imports for any name in _EXTRA_PATCHES
class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Only intercept modules we've declared in _EXTRA_PATCHES
        if fullname not in _EXTRA_PATCHES:
            return None

        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception:
            log("PathFinder.find_spec raised:\n" + traceback.format_exc())
            return None

        if spec is None or spec.loader is None:
            return None

        orig_loader = spec.loader

        # already wrapped?
        if getattr(orig_loader, "_wrapped_by_vllm_auto", False):
            return spec

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                if hasattr(orig_loader, "create_module"):
                    return orig_loader.create_module(spec)
                return None

            def exec_module(self, module):
                # let original loader load module first
                orig_loader.exec_module(module)
                # dispatch patch according to module name
                try:
                    _dispatch_patch_for_module(module)
                except Exception:
                    log("apply_patch_dispatch failed in Loader.exec_module:\n" + traceback.format_exc())

        try:
            setattr(orig_loader, "_wrapped_by_vllm_auto", True)
        except Exception:
            pass

        spec.loader = _Loader()
        log("install loader wrapper for " + fullname)
        return spec


# install finder if not already present
if not any(isinstance(x, _Finder) for x in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
    log("vllm_auto_patch: meta_path finder installed")
