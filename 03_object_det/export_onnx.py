#!/usr/bin/env python3
"""
将YOLOv5/YOLOv8 PyTorch模型导出为ONNX格式
支持动态batch size和输入尺寸
"""

import torch
import argparse
from pathlib import Path
import sys

# 添加ultralytics路径
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics未安装,仅支持标准PyTorch模型")

def export_yolo_to_onnx(
    weights,
    img_size=640,
    batch_size=1,
    dynamic=False,
    simplify=True,
    opset=12,
    output=None
):
    """
    导出YOLO模型到ONNX
    
    Args:
        weights: 模型权重路径(.pt)
        img_size: 输入图像尺寸
        batch_size: 批大小(如果dynamic=True则被忽略)
        dynamic: 是否支持动态batch
        simplify: 是否简化ONNX图
        opset: ONNX opset版本
        output: 输出文件路径
    """
    
    # 确定输出路径
    if output is None:
        output = Path(weights).with_suffix('.onnx')
    
    print(f"导出配置:")
    print(f"  输入模型: {weights}")
    print(f"  输出路径: {output}")
    print(f"  输入尺寸: {img_size}x{img_size}")
    print(f"  批大小: {batch_size}")
    print(f"  动态batch: {dynamic}")
    print(f"  ONNX opset: {opset}")
    
    # 方法1: 使用ultralytics (推荐)
    if ULTRALYTICS_AVAILABLE and (str(weights).endswith('.pt') or 'yolov' in str(weights).lower()):
        print("\n使用ultralytics导出...")
        model = YOLO(weights)
        
        # ultralytics的export方法
        model.export(
            format='onnx',
            imgsz=img_size,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset
        )
        
        print(f"✓ ONNX模型已生成: {output}")
        return str(output)
    
    # 方法2: 标准PyTorch导出
    else:
        print("\n使用PyTorch导出...")
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(weights, map_location=device)
        
        if isinstance(model, dict):
            model = model['model']
        
        model.eval()
        
        # 创建dummy输入
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
        
        # 动态轴配置
        if dynamic:
            dynamic_axes = {
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
        else:
            dynamic_axes = None
        
        # 导出
        torch.onnx.export(
            model,
            dummy_input,
            output,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"✓ ONNX模型已导出")
        
        # 简化ONNX
        if simplify:
            try:
                import onnx
                import onnxsim
                
                print("\n简化ONNX模型...")
                model_onnx = onnx.load(output)
                model_sim, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': [batch_size, 3, img_size[0], img_size[1]]} if not dynamic else None
                )
                onnx.save(model_sim, output)
                print(f"✓ ONNX模型已简化")
                
            except ImportError:
                print("警告: onnx-simplifier未安装,跳过简化")
                print("安装: pip install onnx-simplifier")
    
    # 验证ONNX
    try:
        import onnx
        
        print("\n验证ONNX模型...")
        model_onnx = onnx.load(output)
        onnx.checker.check_model(model_onnx)
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  Graph名称: {model_onnx.graph.name}")
        print(f"  输入:")
        for input_tensor in model_onnx.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in input_tensor.type.tensor_type.shape.dim]
            print(f"    {input_tensor.name}: {shape}")
        print(f"  输出:")
        for output_tensor in model_onnx.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in output_tensor.type.tensor_type.shape.dim]
            print(f"    {output_tensor.name}: {shape}")
        
        print(f"\n✓ ONNX验证通过")
        
    except Exception as e:
        print(f"警告: ONNX验证失败: {e}")
    
    return str(output)

def main():
    parser = argparse.ArgumentParser(description='导出YOLO模型到ONNX')
    parser.add_argument('--weights', type=str, required=True,
                       help='PyTorch模型权重路径(.pt)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批大小')
    parser.add_argument('--dynamic', action='store_true',
                       help='启用动态batch size')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='简化ONNX图')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset版本')
    parser.add_argument('--output', type=str, default=None,
                       help='输出ONNX文件路径')
    
    args = parser.parse_args()
    
    output = export_yolo_to_onnx(
        weights=args.weights,
        img_size=args.img_size,
        batch_size=args.batch_size,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        output=args.output
    )
    
    print(f"\n完成!")
    print(f"ONNX模型: {output}")
    print(f"\n下一步:")
    print(f"  python tensorrt_optimization/build_engine.py --onnx {output}")

if __name__ == '__main__':
    main()
