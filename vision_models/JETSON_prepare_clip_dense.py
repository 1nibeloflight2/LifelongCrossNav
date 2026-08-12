import torch
import tensorrt as trt
import numpy as np
from clip_dense import ClipModel
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

def build_engine(model, input_shape, text=False):

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 8GB
    if not text:
        # for some reason this doesnt work for the text boi
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    # config.set_flag(trt.BuilderFlag.TF32)  # This enables TF32 computation where possible
    # config.set_flag(trt.BuilderFlag.TIMING_CACHE)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=input_shape, opt=input_shape, max=input_shape)
    config.add_optimization_profile(profile)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # input_tensor = network.add_input("input_tensor", trt.DataType.FLOAT, input_shape)
    # profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    # config.add_optimization_profile(profile)

    # Convert PyTorch model to ONNX
    dummy_input = torch.randn(input_shape).cuda() # Image
    #dummy_input = torch.randn(input_shape)*64 # Text
    if text:
        texts=["abcd"]
        dummy_input = model.tokenizer(texts).to("cuda")

    torch.onnx.export(model, dummy_input, "temp.onnx",
                      input_names=["input_tensor"], output_names=["output"], opset_version=17)

    # Parse ONNX file
    parser = trt.OnnxParser(network, logger)
    with open("temp.onnx", 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    engine = builder.build_engine(network, config)
    return engine

def optimize_for_jetson():
    # Load your model
    model = ClipModel('weights/clip.pth', False)
    model.eval().cuda()

    # Example input shape
    im_input_shape = (1, 3, 768, 768) # Image
    model.forward = model.forward_im
    # Build TensorRT engine
    engine = build_engine(model, im_input_shape, False)

    # Save the engine
    with open("trt/clip_model_trt.engine", "wb") as f:
        f.write(engine.serialize())

    print("TensorRT engine saved to clip_model_trt.engine")

    input_shape = (1, 77) # Text
    model.forward= model.forward_text
    engine = build_engine(model, input_shape, True)
        # Save the engine
    with open("trt/clip_text_model_trt.engine", "wb") as f:
        f.write(engine.serialize())

    print("TensorRT engine saved to clip_text_model_trt.engine")




def load_and_test():
    original_model = ClipModel('weights/clip.pth', False)
    original_model.eval().cuda()
    original_model.forward= original_model.forward_im
    # Load the saved TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    
    ### TRT MODEL LOADING
    runtime = trt.Runtime(logger)
    with open("trt/clip_model_trt.engine", "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    #### /TRT MODEL LOADING

    
    # Prepare input data
    input_shape = (1, 3, 640, 640)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(input_data).cuda()
    # Allocate device memory
    d_input = cuda.mem_alloc(input_data.nbytes)
    output = np.empty(engine.get_binding_shape(1), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    stream = cuda.Stream()
    # Warm-up runs
    for _ in range(10):
        # TensorRT warm-up
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        # PyTorch warm-up
        with torch.no_grad():
            _ = original_model(torch_input)
    # Time TensorRT inference
    num_iterations = 20
    trt_start_time = time.time()
    for _ in range(num_iterations):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
    trt_end_time = time.time()
    trt_avg_time = (trt_end_time - trt_start_time) / num_iterations

    # Time PyTorch inference
    torch.cuda.synchronize()
    pytorch_start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = original_model(torch_input)
    torch.cuda.synchronize()
    pytorch_end_time = time.time()
    pytorch_avg_time = (pytorch_end_time - pytorch_start_time) / num_iterations

    print(f"TensorRT engine inference successful")
    print(f"TensorRT output shape: {output.shape}")
    print(f"TensorRT average inference time: {trt_avg_time * 1000:.2f} ms")
    print(f"PyTorch average inference time: {pytorch_avg_time * 1000:.2f} ms")
    print(f"Speed-up: {pytorch_avg_time / trt_avg_time:.2f}x")
    
def load_and_test_trt_only():
    trt_model = ClipModel('weights/clip.pth', False)
    trt_model.eval().cuda()
    # Load the saved TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    
    # ### TRT MODEL LOADING
    # runtime = trt.Runtime(logger)
    # with open("trt/clip_model_trt.engine", "rb") as f:
        # engine = runtime.deserialize_cuda_engine(f.read())
# 
    # context = engine.create_execution_context()
    # #### /TRT MODEL LOADING

    
    # Prepare input data
    input_shape = (1, 3, 640, 640)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    # torch_input = torch.from_numpy(input_data).cuda()
    # # Allocate device memory
    # d_input = cuda.mem_alloc(input_data.nbytes)
    #output = np.empty(engine.get_binding_shape(1), dtype=np.float32)
    #d_output = cuda.mem_alloc(output.nbytes)
    stream = cuda.Stream()
    # Warm-up runs
    for _ in range(10):
        # TensorRT warm-up
        #output = trt_model(input_data)
        print("warmup")
        output = trt_model.get_image_features(input_data)
        # cuda.memcpy_htod_async(d_input, input_data, stream)
        # context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # cuda.memcpy_dtoh_async(output, d_output, stream)
        # stream.synchronize()

        # # PyTorch warm-up
        # with torch.no_grad():
            # _ = original_model(torch_input)
    # Time TensorRT inference
    num_iterations = 20
    trt_start_time = time.time()
    for _ in range(num_iterations):
        print("actual runs")
        output = trt_model.get_image_features(input_data)
        #trt_model(input_data)
    trt_end_time = time.time()
    trt_avg_time = (trt_end_time - trt_start_time) / num_iterations

    # Time PyTorch inference
    # torch.cuda.synchronize()
    # pytorch_start_time = time.time()
    # for _ in range(num_iterations):
        # with torch.no_grad():
            # _ = original_model(torch_input)
    # torch.cuda.synchronize()
    # pytorch_end_time = time.time()
    # pytorch_avg_time = (pytorch_end_time - pytorch_start_time) / num_iterations

    print(f"TensorRT engine inference successful")
    print(f"TensorRT output shape: {output.shape}")
    print(f"TensorRT average inference time: {trt_avg_time * 1000:.2f} ms")
    # print(f"PyTorch average inference time: {pytorch_avg_time * 1000:.2f} ms")
    # print(f"Speed-up: {pytorch_avg_time / trt_avg_time:.2f}x")



if __name__ == "__main__":
    optimize_for_jetson()
    # load_and_test()
    #load_and_test_trt_only()
