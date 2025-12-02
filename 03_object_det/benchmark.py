#!/usr/bin/env python3
"""
TensorRT引擎性能测试
对比不同精度的速度和精度
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import argparse
from pathlib import Path
import cv2

class TRTBenchmark:
    def __init__(self, engine_path):
        """加载TensorRT引擎"""
        print(f"加载引擎: {engine_path}")
        
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        
        print(f"  输入shape: {self.input_shape}")
        print(f"  输出shape: {self.output_shape}")
        
        # 分配GPU内存
        self.d_input = cuda.mem_alloc(
            np.prod(self.input_shape) * np.float32().itemsize
        )
        self.d_output = cuda.mem_alloc(
            np.prod(self.output_shape) * np.float32().itemsize
        )
        
        # 预热
        dummy = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(10):
            self.infer(dummy)
        
        print("✓ 引擎已加载并预热")
    
    def infer(self, input_data):
        """执行推理"""
        # 拷贝到GPU
        cuda.memcpy_htod(self.d_input, input_data)
        
        # 推理
        self.context.execute_v2([int(self.d_input), int(self.d_output)])
        
        # 拷贝回CPU
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)
        
        return output
    
    def benchmark_latency(self, num_iterations=100):
        """测试延迟"""
        print(f"\n测试延迟 ({num_iterations}次迭代)...")
        
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.infer(dummy_input)
            latencies.append((time.time() - start) * 1000)
        
        latencies = np.array(latencies)
        
        print(f"  平均延迟: {latencies.mean():.2f} ms")
        print(f"  中位数: {np.median(latencies):.2f} ms")
        print(f"  标准差: {latencies.std():.2f} ms")
        print(f"  最小值: {latencies.min():.2f} ms")
        print(f"  最大值: {latencies.max():.2f} ms")
        print(f"  FPS: {1000 / latencies.mean():.1f}")
        
        return latencies.mean()
    
    def benchmark_throughput(self, duration=10):
        """测试吞吐量"""
        print(f"\n测试吞吐量 (运行{duration}秒)...")
        
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            _ = self.infer(dummy_input)
            count += 1
        
        elapsed = time.time() - start_time
        throughput = count / elapsed
        
        print(f"  总帧数: {count}")
        print(f"  吞吐量: {throughput:.1f} FPS")
        print(f"  平均延迟: {1000/throughput:.2f} ms")
        
        return throughput

def compare_engines(engine_paths):
    """对比多个引擎的性能"""
    print("="*60)
    print("TensorRT引擎性能对比")
    print("="*60)
    
    results = {}
    
    for engine_path in engine_paths:
        print(f"\n【{Path(engine_path).name}】")
        
        try:
            benchmark = TRTBenchmark(engine_path)
            latency = benchmark.benchmark_latency(num_iterations=100)
            throughput = benchmark.benchmark_throughput(duration=10)
            
            results[engine_path] = {
                'latency': latency,
                'throughput': throughput
            }
        
        except Exception as e:
            print(f"  错误: {e}")
            results[engine_path] = None
    
    # 打印对比表
    print("\n" + "="*60)
    print("对比汇总")
    print("="*60)
    print(f"{'引擎':<30} {'延迟(ms)':<15} {'吞吐量(FPS)':<15}")
    print("-"*60)
    
    for engine_path, result in results.items():
        if result:
            print(f"{Path(engine_path).name:<30} "
                  f"{result['latency']:<15.2f} "
                  f"{result['throughput']:<15.1f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='TensorRT性能测试')
    parser.add_argument('--engines', type=str, nargs='+', required=True,
                       help='TensorRT引擎路径(可多个)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='延迟测试迭代次数')
    parser.add_argument('--duration', type=int, default=10,
                       help='吞吐量测试持续时间(秒)')
    
    args = parser.parse_args()
    
    if len(args.engines) == 1:
        # 单个引擎详细测试
        benchmark = TRTBenchmark(args.engines[0])
        benchmark.benchmark_latency(args.iterations)
        benchmark.benchmark_throughput(args.duration)
    else:
        # 多个引擎对比
        compare_engines(args.engines)

if __name__ == '__main__':
    main()
