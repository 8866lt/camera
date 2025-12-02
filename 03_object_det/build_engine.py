#!/usr/bin/env python3
"""
将ONNX模型构建为TensorRT引擎
支持FP32/FP16/INT8精度
支持动态输入尺寸
"""

import tensorrt as trt
import argparse
from pathlib import Path
import sys
import numpy as np

# TensorRT日志级别
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class EngineBuilder:
    def __init__(self, onnx_path, verbose=False):
        """
        初始化引擎构建器
        
        Args:
            onnx_path: ONNX模型路径
            verbose: 是否显示详细信息
        """
        self.onnx_path = onnx_path
        
        if verbose:
            TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
        
        # 创建builder
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # 解析ONNX
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)
        
        print(f"TensorRT版本: {trt.__version__}")
    
    def parse_onnx(self):
        """解析ONNX模型"""
        print(f"\n解析ONNX: {self.onnx_path}")
        
        with open(self.onnx_path, 'rb') as f:
            if not self.parser.parse(f.read()):
                for error in range(self.parser.num_errors):
                    print(f"ERROR: {self.parser.get_error(error)}")
                sys.exit(1)
        
        print("✓ ONNX解析成功")
        
        # 打印网络信息
        print(f"\n网络信息:")
        print(f"  输入数量: {self.network.num_inputs}")
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            print(f"    {input_tensor.name}: {input_tensor.shape}")
        
        print(f"  输出数量: {self.network.num_outputs}")
        for i in range(self.network.num_outputs):
            output_tensor = self.network.get_output(i)
            print(f"    {output_tensor.name}: {output_tensor.shape}")
    
    def set_precision(self, fp16=False, int8=False, calibrator=None):
        """
        设置精度模式
        
        Args:
            fp16: 是否使用FP16
            int8: 是否使用INT8
            calibrator: INT8校准器
        """
        if fp16:
            if self.builder.platform_has_fast_fp16:
                self.config.set_flag(trt.BuilderFlag.FP16)
                print("✓ 启用FP16精度")
            else:
                print("警告: 平台不支持FP16,使用FP32")
        
        if int8:
            if self.builder.platform_has_fast_int8:
                self.config.set_flag(trt.BuilderFlag.INT8)
                
                if calibrator is None:
                    print("警告: INT8需要校准器,将使用FP16")
                    self.config.set_flag(trt.BuilderFlag.FP16)
                else:
                    self.config.int8_calibrator = calibrator
                    print("✓ 启用INT8精度")
            else:
                print("警告: 平台不支持INT8,使用FP16")
                self.config.set_flag(trt.BuilderFlag.FP16)
    
    def set_workspace(self, size_gb=4):
        """设置GPU工作空间"""
        workspace_size = size_gb * (1 << 30)  # GB转字节
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 
            workspace_size
        )
        print(f"✓ 设置工作空间: {size_gb} GB")
    
    def set_dynamic_shape(self, input_name, min_shape, opt_shape, max_shape):
        """
        设置动态输入尺寸
        
        Args:
            input_name: 输入张量名称
            min_shape: 最小尺寸 (batch, channels, height, width)
            opt_shape: 优化尺寸
            max_shape: 最大尺寸
        """
        profile = self.builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        self.config.add_optimization_profile(profile)
        
        print(f"\n动态输入尺寸:")
        print(f"  最小: {min_shape}")
        print(f"  优化: {opt_shape}")
        print(f"  最大: {max_shape}")
    
    def build(self, output_path):
        """构建并保存TensorRT引擎"""
        print(f"\n开始构建TensorRT引擎...")
        print("这可能需要几分钟,请耐心等待...")
        
        # 构建序列化引擎
        serialized_engine = self.builder.build_serialized_network(
            self.network, 
            self.config
        )
        
        if serialized_engine is None:
            print("ERROR: 引擎构建失败")
            sys.exit(1)
        
        # 保存引擎
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"\n✓ TensorRT引擎已保存: {output_path}")
        
        # 获取引擎大小
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  文件大小: {size_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='构建TensorRT引擎')
    parser.add_argument('--onnx', type=str, required=True,
                       help='ONNX模型路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出引擎路径(.engine)')
    parser.add_argument('--fp16', action='store_true',
                       help='启用FP16精度')
    parser.add_argument('--int8', action='store_true',
                       help='启用INT8精度(需要校准)')
    parser.add_argument('--calib-images', type=str, default=None,
                       help='INT8校准图像目录')
    parser.add_argument('--calib-cache', type=str, default='calibration.cache',
                       help='INT8校准缓存文件')
    parser.add_argument('--workspace', type=float, default=4,
                       help='GPU工作空间大小(GB)')
    parser.add_argument('--dynamic', action='store_true',
                       help='启用动态输入尺寸')
    parser.add_argument('--min-size', type=int, default=320,
                       help='最小输入尺寸(动态模式)')
    parser.add_argument('--opt-size', type=int, default=640,
                       help='优化输入尺寸(动态模式)')
    parser.add_argument('--max-size', type=int, default=1280,
                       help='最大输入尺寸(动态模式)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        onnx_path = Path(args.onnx)
        if args.fp16:
            suffix = '_fp16.engine'
        elif args.int8:
            suffix = '_int8.engine'
        else:
            suffix = '_fp32.engine'
        args.output = str(onnx_path.with_suffix(suffix))
    
    # 创建构建器
    builder = EngineBuilder(args.onnx, verbose=args.verbose)
    
    # 解析ONNX
    builder.parse_onnx()
    
    # 设置精度
    calibrator = None
    if args.int8:
        if args.calib_images:
            from calibrator import ImageCalibrator
            calibrator = ImageCalibrator(
                calib_images=args.calib_images,
                cache_file=args.calib_cache,
                input_shape=(args.opt_size, args.opt_size)
            )
        else:
            print("警告: INT8模式需要校准图像,使用--calib-images指定")
            args.int8 = False
            args.fp16 = True
    
    builder.set_precision(fp16=args.fp16, int8=args.int8, calibrator=calibrator)
    
    # 设置工作空间
    builder.set_workspace(args.workspace)
    
    # 动态尺寸
    if args.dynamic:
        builder.set_dynamic_shape(
            'images',
            min_shape=(1, 3, args.min_size, args.min_size),
            opt_shape=(1, 3, args.opt_size, args.opt_size),
            max_shape=(1, 3, args.max_size, args.max_size)
        )
    
    # 构建引擎
    builder.build(args.output)
    
    print(f"\n完成!")
    print(f"TensorRT引擎: {args.output}")
    print(f"\n下一步:")
    print(f"  python yolo_detection.py --model {args.output} --source 0")

if __name__ == '__main__':
    main()
