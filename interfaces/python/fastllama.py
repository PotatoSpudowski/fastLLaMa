import os
import ctypes
from enum import Enum
import multiprocessing
from typing import Any, Callable, List, Optional, Type, Union, cast
import signal

LIBRARY_NAME='pyfastllama.so'

def get_library_path(*args) -> str:
    return os.path.join(*args, LIBRARY_NAME)

class Logger:
    """
    Logger class for reporting messages.
    """
    def log_info(self, func_name: str, message: str) -> None:
        """
        Logs informational messages.

        :param func_name: Name of the function that generated the log message.
        :param message: The log message.
        """
        print(f"[Info]: Func('{func_name}') {message}", flush=True, end='')
    
    def log_err(self, func_name: str, message: str) -> None:
        """
        Logs error messages.

        :param func_name: Name of the function that generated the log message.
        :param message: The log message.
        """
        print(f"[Error]: Func('{func_name}') {message}", flush=True, end='')
    
    def log_warn(self, func_name: str, message: str) -> None:
        """
        Logs warning messages.

        :param func_name: Name of the function that generated the log message.
        :param message: The log message.
        """
        print(f"[Warn]: Func('{func_name}') {message}", flush=True, end='')
    
    def reset(self) -> None:
        """
        Resets the logger state.
        """
        return None

class ModelKind(Enum):
    """
    Enumeration of supported model kinds.
    """
    LLAMA_7B = "LLAMA-7B"
    LLAMA_13B = "LLAMA-13B"
    LLAMA_30B = "LLAMA-30B"
    LLAMA_65B = "LLAMA-65B"
    ALPACA_LORA_7B = "ALPACA-LORA-7B"
    ALPACA_LORA_13B = "ALPACA-LORA-13B"
    ALPACA_LORA_30B = "ALPACA-LORA-30B"
    ALPACA_LORA_65B = "ALPACA-LORA-65B"

C_LLAMA_LOGGER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
C_LLAMA_LOGGER_RESET_FUNC = ctypes.CFUNCTYPE(None)

class c_llama_logger(ctypes.Structure):
    """
    C-compatible logger structure.
    """
    _fields_ = [
        ('log', C_LLAMA_LOGGER_FUNC),
        ('log_err', C_LLAMA_LOGGER_FUNC),
        ('log_warn', C_LLAMA_LOGGER_FUNC),
        ('reset', C_LLAMA_LOGGER_RESET_FUNC)
    ]

class llama_array_view_f(ctypes.Structure):
    """
    C-compatible array view structure for floats.
    """
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('size', ctypes.c_size_t),
    ]

class c_llama_model_context_args(ctypes.Structure):
    """
    C-compatible model context arguments structure.
    """
    _fields_ = [
        ('embedding_eval_enabled', ctypes.c_bool),
        ('should_get_all_logits', ctypes.c_bool),
        ('seed', ctypes.c_int),
        ('n_keep', ctypes.c_int),
        ('n_ctx', ctypes.c_int),
        ('n_threads', ctypes.c_int),
        ('n_batch', ctypes.c_int),
        ('last_n_tokens', ctypes.c_size_t),
        ('allocate_extra_mem', ctypes.c_size_t),
        ('logger', c_llama_logger)
    ]

class c_llama_model_context(ctypes.Structure):
    """
    C-compatible model context structure.
    """
    pass

c_llama_model_context_ptr = ctypes.POINTER(c_llama_model_context)

def make_c_logger_func(func: Callable[[str, str], None]) -> Any:
    """
    Creates a C-compatible logger function from a Python callable.

    :param func: Python callable to be converted to a C-compatible logger function.
    :return: C-compatible logger function.
    """
    def c_logger_func(func_name: ctypes.c_char_p, func_name_len: ctypes.c_int, message: ctypes.c_char_p, message_len: ctypes.c_int) -> None:
        func(ctypes.string_at(func_name, int(func_name_len)).decode('utf-8'), ctypes.string_at(message, int(message_len)).decode('utf-8'))
    return C_LLAMA_LOGGER_FUNC(c_logger_func)

def make_c_logger_reset_func(func: Callable[[], None]) -> Any:
    """
    Creates a C-compatible logger reset function from a Python callable.

    :param func: Python callable to be converted to a C-compatible logger reset function.
    :return: C-compatible logger reset function.
    """
    def c_logger_func() -> None:
        func()
    return C_LLAMA_LOGGER_RESET_FUNC(c_logger_func)

class Model:
    """
    Model class for interacting with the language model.
    """
    def __init__(
        self,
        id: Union[str, ModelKind], 
        path: str, 
        num_threads: int = multiprocessing.cpu_count(), 
        n_ctx: int = 512, 
        last_n_size: int = 64, 
        seed: int = 0, 
        tokens_to_keep: int = 200,
        n_batch: int = 16, 
        should_get_all_logits: bool = False,
        embedding_eval_enabled: bool = False,
        allocate_extra_mem: int = 0,
        logger: Optional[Logger] = None, 
        library_path = get_library_path('build', 'interfaces','python')
        ):
        """
        Initializes a new model instance.

        :param id: Model identifier. Can be a string or a ModelKind enum value.
        :param path: Path to the model file.
        :param num_threads: Number of threads to use during model evaluation. Default is the number of CPU cores.
        :param n_ctx: Size of the memory context to use. Default is 512.
        :param last_n_size: Number of tokens the model can remember. Default is 64.
        :param seed: Random number seed to be used in the model. Default is 0.
        :param tokens_to_keep: Number of tokens to keep when tokens are removed from the buffer to save memory. Default is 200.
        :param n_batch: Size of the token batch that will be processed at a given time. Default is 16.
        :param should_get_all_logits: Flag to indicate if all logit values should be returned. Default is False.
        :param embedding_eval_enabled: Flag to enable embedding evaluation. Default is False.
        :param allocate_extra_mem: Amount of extra memory to allocate. Default is 0.
        :param logger: Logger instance to be used for reporting messages. Default is None.
        :param library_path: Path to the library file. Default is the result of get_library_path('build', 'interfaces','python').
        """

        self.lib = ctypes.cdll.LoadLibrary(library_path)

        signal_handler_fn = self.lib.llama_handle_signal
        signal_handler_fn.argtypes = [ctypes.c_int]

        signal.signal(signal.SIGINT, lambda sig_num, _frame: signal_handler_fn(sig_num))
        signal.siginterrupt(signal.SIGINT, True)

        normalized_id: str = cast(str, id.value if type(id) == ModelKind else id)

        ctx_args = self.__get_default_ctx_args__()

        ctx_args.seed = seed
        ctx_args.n_keep = tokens_to_keep
        ctx_args.n_ctx = n_ctx
        ctx_args.n_threads = num_threads
        ctx_args.n_batch = n_batch
        ctx_args.last_n_tokens = last_n_size
        ctx_args.embedding_eval_enabled = embedding_eval_enabled
        ctx_args.should_get_all_logits = should_get_all_logits
        ctx_args.allocate_extra_mem = allocate_extra_mem

        if logger is not None:
            temp_logger = c_llama_logger()
            temp_logger.log = make_c_logger_func(logger.log_info)
            temp_logger.log_err = make_c_logger_func(logger.log_err)
            temp_logger.log_warn = make_c_logger_func(logger.log_warn)
            temp_logger.reset = make_c_logger_reset_func(logger.reset)
            ctx_args.logger = temp_logger

        self.ctx = self.__create_model_ctx__(ctx_args)

        load_fn = self.lib.llama_load_model_str
        load_fn.restype = ctypes.c_bool
        load_fn.argtypes = [c_llama_model_context_ptr, ctypes.c_char_p, ctypes.c_char_p]
        res = bool(load_fn(self.ctx, bytes(normalized_id, 'utf-8'), bytes(path, 'utf-8')))
        if not res:
            raise RuntimeError("Unable to load model")

    def __get_default_ctx_args__(self) -> c_llama_model_context_args:
        """
        Retrieves the default model context arguments.

        :return: Default model context arguments.
        """
        fn = self.lib.llama_create_default_context_args
        fn.restype = c_llama_model_context_args
        return fn()
    
    def __create_model_ctx__(self, ctx_args: c_llama_model_context_args):
        """
        Creates a new model context.

        :param ctx_args: Model context arguments.
        :return: Model context pointer.
        """
        fn = self.lib.llama_create_context
        fn.restype = c_llama_model_context_ptr
        fn.argtypes = [c_llama_model_context_args]
        return fn(ctx_args)
    
    def save_state(self, filepath: str) -> bool:
        """
        Saves the current model state to a file.

        :param filepath: Path to the file where the model state will be saved.
        :return: True if successful, False otherwise.
        """
        fn = self.lib.llama_save_state
        fn.argtypes = [c_llama_model_context_ptr, ctypes.c_char_p]
        fn.restype = ctypes.c_bool
        return bool(fn(self.ctx, bytes(filepath, 'utf-8')))
    
    def load_state(self, filepath: str) -> bool:
        """
        Loads a model state from a file.

        :param filepath: Path to the file containing the model state to load.
        :return: True if successful, False otherwise.
        """
        fn = self.lib.llama_load_state
        fn.argtypes = [c_llama_model_context_ptr, ctypes.c_char_p]
        fn.restype = ctypes.c_bool
        return bool(fn(self.ctx, bytes(filepath, 'utf-8')))

    def ingest(self, prompt: str, is_system_prompt: bool = False) -> bool:
        """
        Ingests a prompt into the model.

        :param prompt: The prompt to be ingested.
        :param is_system_prompt: Flag to indicate if the prompt is a system prompt. Default is False.
        :return: True if successful, False otherwise.
        """
        if is_system_prompt:
            ingest_fn = self.lib.llama_ingest_system_prompt
        else:
            ingest_fn = self.lib.llama_ingest

        ingest_fn.argtypes = [c_llama_model_context_ptr, ctypes.c_char_p]
        ingest_fn.restype = ctypes.c_bool
        return bool(ingest_fn(self.ctx, bytes(prompt, 'utf-8')))
    
    def generate(
            self,
            streaming_fn=Callable[[str], None],
            num_tokens: int = 100, 
            top_k: int = 40, 
            top_p: float = .95, 
            temp: float = .8, 
            repeat_penalty: float = 1.0, 
            stop_words: List[str] = [], 
        ) -> bool:
        """
        Generates text using the model.

        :param streaming_fn: Function to be called with the generated text.
        :param num_tokens: Maximum number of tokens to be generated by the model. Default is 100.
        :param top_k: Controls the diversity by limiting the selection to the top k highest probability tokens. Default is 40.
        :param top_p: Filters out tokens based on cumulative probability, further refining diversity. Default is 0.95.
        :param temp: Adjusts the sampling temperature, influencing creativity and randomness. Default is 0.8.
        :param repeat_penalty: Penalizes repeated tokens to reduce redundancy in generated text. Default is 1.0.
        :param stop_words: List of words that will stop the generate function when encountered in the token buffer. Default is an empty list.
        :return: True if successful, False otherwise.
        """
        def callback_fn(token: ctypes.c_char_p, len: ctypes.c_int):
            arr = ctypes.string_at(token, int(len))
            streaming_fn(arr.decode('utf-8'))
        stop_words_ptr_type = (ctypes.c_char_p * len(stop_words))
        stop_words_fn = self.lib.llama_set_stop_words
        stop_words_fn.restype = ctypes.c_bool
        stop_words_fn.argtypes = cast(List[Type[Any]], [c_llama_model_context_ptr, stop_words_ptr_type, ctypes.c_size_t])

        stop_words_fn(self.ctx, stop_words_ptr_type(*[bytes(s, 'utf-8') for s in stop_words]), len(stop_words))

        generate_fn = self.lib.llama_generate
        ctype_callback_fn = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int)
        generate_fn.argtypes = [
            c_llama_model_context_ptr,
            ctype_callback_fn,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float
        ]
        generate_fn.restype = ctypes.c_bool
        return bool(generate_fn(
            self.ctx,
            ctype_callback_fn(callback_fn),
            num_tokens,
            top_k,
            top_p,
            temp,
            repeat_penalty,
        ))
    
    def perplexity(self, prompt: str) -> Optional[float]:
        """
        Calculates the perplexity of a given prompt.

        :param prompt: The prompt to calculate perplexity for.
        :return: Perplexity value if successful, None otherwise.
        """
        per_fun = self.lib.llama_perplexity
        per_fun.restype = ctypes.c_float
        per_fun.argtypes = [c_llama_model_context_ptr, ctypes.c_char_p]
        res = float(per_fun(self.ctx, bytes(prompt, 'utf-8')))
        if res < 0:
            return None
        return res
    
    def get_embeddings(self) -> List[float]:
        """
        Retrieves the embeddings of the model.

        :return: List of embeddings.
        """
        getter = self.lib.llama_get_embeddings
        getter.restype = llama_array_view_f
        getter.argtypes = [c_llama_model_context_ptr]
        res: llama_array_view_f = getter(self.ctx)
        return res.data[:int(res.size)]
    
    def get_logits(self) -> List[float]:
        """
        Retrieves the logits of the model.

        :return: List of logits.
        """
        getter = self.lib.llama_get_logits
        getter.restype = llama_array_view_f
        getter.argtypes = [c_llama_model_context_ptr]
        res: llama_array_view_f = getter(self.ctx)
        return res.data[:int(res.size)]
    
    def attach_lora(self, filepath: str) -> bool:
        """
        Attaches a Lora model to the current model.

        :param filepath: Path to the Lora model file.
        :return: True if successful, False otherwise.
        """
        fn = self.lib.llama_attach_lora
        fn.argtypes = [c_llama_model_context_ptr, ctypes.c_char_p]
        fn.restype = ctypes.c_bool
        return bool(fn(self.ctx, bytes(filepath, 'utf-8')))

    def detach_lora(self) -> bool:
        """
        Detaches the Lora model from the current model.

        :return: True if successful, False otherwise.
        """
        fn = self.lib.llama_detach_lora
        fn.argtypes = [c_llama_model_context_ptr]
        fn.restype = ctypes.c_bool
        return bool(fn(self.ctx))

    def reset(self) -> bool:
        """
        Resets the model.

        :return: True if successful, False otherwise.
        """
        fn = self.lib.llama_reset_model
        fn.argtypes = [c_llama_model_context_ptr]
        fn.restype = ctypes.c_bool
        return bool(fn(self.ctx))
    
    def __del__(self):
        """
        Destructor for the Model class.
        """
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.siginterrupt(signal.SIGINT, False)
        lib = self.lib
        ctx = self.ctx
        free_fn = lib.llama_free_context
        free_fn.argtypes = [ctypes.POINTER(c_llama_model_context)]
        free_fn(ctx)
        pass
