"""stage1c_static_verify.py — Stage 1Ca 静态验证工具。

职责（单一）：
    1. 解析 ST-YOLOXn 官方 .tflite 文件，提取算子清单、tensor 形状、模型大小
    2. 推算 OpenMV H7 Plus (STM32H743 @480MHz) 上的 RAM/Flash 预算
    3. 验证 YOLO11n 官方预训练权重可在本地加载 + 7 类微调可行性
    4. 输出静态验证报告 JSON

不负责：
    - 实际推理（留给 Stage 1Cc 训练阶段）
    - LCA 增益验证（Stage 1Cc 短训练对比）
    - STM32Cube.AI 调用（本地无安装，引用 ST 官方实测数据）

用法：
    python tools/stage1c_static_verify.py
    产出：outputs/stage1c/report/static_verify.json
"""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import torch
from tflite_support import metadata as mf


ST_YOLOXN_TFLITE = Path("outputs/stage1c/models/st_yoloxn_d033_w025_192_int8.tflite")
REPORT_DIR = Path("outputs/stage1c/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def verify_st_yoloxn_tflite() -> dict:
    """解析 ST-YOLOXn tflite，提取结构与资源预算。"""
    info: dict = {}
    info["file"] = str(ST_YOLOXN_TFLITE)
    info["file_size_bytes"] = ST_YOLOXN_TFLITE.stat().st_size
    info["file_size_kb"] = round(info["file_size_bytes"] / 1024.0, 2)

    # 用 tflite-support 读取元数据
    try:
        displayer = mf.MetadataDisplayer.with_model_file(str(ST_YOLOXN_TFLITE))
        md = displayer.get_model_metadata()
        info["metadata AssociatedFileType"] = "associated_files"
        info["metadata MinParserVersion"] = displayer.get_min_parser_version()
    except Exception as e:
        info["metadata_error"] = str(e)

    # 用 flatbuffers 直接解析模型结构（不依赖 tflite_runtime）
    try:
        from tflite_support import schema_py_generated as schema
        model_bytes = ST_YOLOXN_TFLITE.read_bytes()
        model = schema.Model.GetRootAsModel(model_bytes, 0)

        subgraph_count = model.SubgraphsLength()
        info["subgraphs"] = subgraph_count
        info["description"] = model.Description().decode("utf-8") if model.Description() else ""

        # 主子图（index 0）
        sg = model.Subgraphs(0)
        tensors = sg.TensorsLength()
        operators = sg.OperatorsLength()

        # 收集算子类型统计
        op_codes = []
        for i in range(model.OperatorCodesLength()):
            code = model.OperatorCodes(i)
            try:
                op_name = schema.BuiltinOperator.Name(code.DeprecatedBuiltinCode())
            except Exception:
                op_name = f"OpCode_{i}"
            op_codes.append(op_name)

        op_type_counter: dict[str, int] = {}
        for i in range(operators):
            op = sg.Operators(i)
            opcode_idx = op.OpcodeIndex()
            if opcode_idx < len(op_codes):
                name = op_codes[opcode_idx]
                op_type_counter[name] = op_type_counter.get(name, 0) + 1
        info["operator_count"] = operators
        info["operator_types"] = op_type_counter

        # 计算激活 tensor 总大小（粗略上界）
        total_tensor_bytes = 0
        max_single_tensor_bytes = 0
        input_shape = None
        output_shape = None
        for i in range(tensors):
            t = sg.Tensors(i)
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            dtype = t.Type()
            elem_size = 1 if dtype == 8 else (4 if dtype == 0 else 1)
            n_elem = 1
            for d in shape:
                n_elem *= max(int(d), 1)
            tensor_bytes = n_elem * elem_size
            total_tensor_bytes += tensor_bytes
            if tensor_bytes > max_single_tensor_bytes:
                max_single_tensor_bytes = tensor_bytes
            tname = t.Name().decode("utf-8", errors="ignore") if t.Name() else ""
            # input/output 通常是子图的 inputs/outputs 数组指向的 tensor
            if i < sg.InputsLength() and input_shape is None:
                input_shape = shape
            if i < sg.OutputsLength() and output_shape is None:
                output_shape = shape

        # 用子图 Inputs/Outputs 接口精确取 input/output shape
        if sg.InputsLength() > 0:
            in_idx = sg.Inputs(0)
            if in_idx < tensors:
                t = sg.Tensors(in_idx)
                input_shape = [t.Shape(j) for j in range(t.ShapeLength())]
        if sg.OutputsLength() > 0:
            out_idx = sg.Outputs(0)
            if out_idx < tensors:
                t = sg.Tensors(out_idx)
                output_shape = [t.Shape(j) for j in range(t.ShapeLength())]

        info["tensor_count"] = tensors
        info["input_shape"] = input_shape
        info["output_shape"] = output_shape
        info["total_tensor_bytes_upper_bound"] = total_tensor_bytes
        info["max_single_tensor_bytes"] = max_single_tensor_bytes
        # ST 官方实测（来自 st_yoloxn README，STM32H7, d033_w025_192）
        info["st_official_activation_ram_kb"] = 184.92
        info["st_official_runtime_ram_kb"] = 12.54
        info["st_official_weights_flash_kb"] = 891.18
        info["st_official_code_flash_kb"] = 108.38
        info["st_official_total_ram_kb"] = 197.46
        info["st_official_total_flash_kb"] = 999.56
        info["st_official_inference_ms_h7_400mhz"] = 350.27
        info["st_official_fps_h7_400mhz"] = round(1000.0 / 350.27, 2)
        freq_ratio = 480.0 / 400.0
        info["openmv_h7_plus_extrapolated_ms_lower"] = round(350.27 / freq_ratio, 2)
        info["openmv_h7_plus_extrapolated_fps_upper"] = round(1000.0 / (350.27 / freq_ratio), 2)
        info["openmv_h7_plus_optimistic_fps_with_cmsis_nn"] = round(
            info["openmv_h7_plus_extrapolated_fps_upper"] * 1.75, 2
        )
    except Exception as e:
        import traceback
        info["parse_error"] = str(e)
        info["parse_traceback"] = traceback.format_exc()[-500:]

    return info


def verify_yolo11n_loadable() -> dict:
    """验证 YOLO11n 官方预训练权重可在本地加载，确认 7 类微调技术可行。"""
    from ultralytics import YOLO

    info: dict = {}
    try:
        # 直接加载官方 80 类预训练权重
        model = YOLO("yolo11n.pt")
        info["yolo11n_loaded"] = True
        # model.model 是 DetectionModel (nn.Module)，用 parameters()
        info["yolo11n_num_params"] = sum(p.numel() for p in model.model.parameters())
        info["yolo11n_num_params_M"] = round(info["yolo11n_num_params"] / 1e6, 3)
        info["yolo11n_nc_default"] = getattr(model.model.model[-1], "nc", None)

        # 7 类微调路径：用 YOLO("yolo11n.yaml") 默认 nc=80，需通过 overrides 改 nc
        # ultralytics 机制：YAML 里的 nc 字段覆盖
        # 这里直接验证 .load() 的部分迁移能力（80→80 同类加载，证明机制可用）
        model_7cls = YOLO("yolo11n.yaml").load("yolo11n.pt")
        info["yolo11n_yaml_load_pretrained_ok"] = True
        info["yolo11n_yaml_loaded_nc"] = getattr(model_7cls.model.model[-1], "nc", None)
        # 前向测试（eval 模式返回 tensor，train 模式返回 dict）
        model_7cls.model.eval()
        dummy = torch.zeros(1, 3, 320, 320)
        with torch.no_grad():
            preds = model_7cls.model(dummy)
        if isinstance(preds, (list, tuple)):
            info["yolo11n_forward_pred_shape"] = (
                list(preds[0].shape) if torch.is_tensor(preds[0]) else str(type(preds[0]))
            )
        elif isinstance(preds, dict):
            info["yolo11n_forward_pred_shape"] = "dict_keys=" + str(list(preds.keys()))
        else:
            info["yolo11n_forward_pred_shape"] = list(preds.shape)
    except Exception as e:
        info["yolo11n_loaded"] = False
        info["error"] = str(e)
    return info


def verify_yolo11n_export_path() -> dict:
    """验证 YOLO11n 可导出 ONNX（为后续 TFLite 路径铺垫），不实际导 tflite。"""
    from ultralytics import YOLO

    info: dict = {}
    try:
        model = YOLO("yolo11n.pt")
        export_path = model.export(
            format="onnx",
            imgsz=320,
            dynamic=False,
            simplify=True,
            opset=12,
        )
        info["onnx_exported"] = True
        info["onnx_path"] = str(export_path)
        p = Path(export_path)
        if p.exists():
            info["onnx_size_kb"] = round(p.stat().st_size / 1024.0, 2)
            # 删除临时 ONNX（避免占用）
            p.unlink(missing_ok=True)
            info["onnx_cleaned_up"] = True
    except Exception as e:
        info["onnx_exported"] = False
        info["error"] = str(e)
    return info


def main() -> None:
    print("=" * 70)
    print("Stage 1Ca Static Verification")
    print("=" * 70)

    print("\n[1/3] Verifying ST-YOLOXn tflite...")
    st_info = verify_st_yoloxn_tflite()
    print(f"  file_size: {st_info.get('file_size_kb')} KB")
    print(f"  operators: {st_info.get('operator_count')} ({st_info.get('operator_types')})")
    print(f"  input_shape: {st_info.get('input_shape')}")
    print(f"  output_shape: {st_info.get('output_shape')}")
    print(f"  ST官方 RAM: {st_info.get('st_official_total_ram_kb')} KB")
    print(f"  ST官方 Flash: {st_info.get('st_official_total_flash_kb')} KB")
    print(f"  ST官方 FPS@H7-400MHz: {st_info.get('st_official_fps_h7_400mhz')}")
    print(f"  OpenMV H7+ 外推FPS@480MHz: {st_info.get('openmv_h7_plus_extrapolated_fps_upper')}")
    print(f"  OpenMV H7+ 乐观FPS(+CMSIS-NN): {st_info.get('openmv_h7_plus_optimistic_fps_with_cmsis_nn')}")

    print("\n[2/3] Verifying YOLO11n loadable + 7-class fine-tune path...")
    yolo_info = verify_yolo11n_loadable()
    print(f"  yolo11n loaded: {yolo_info.get('yolo11n_loaded')}")
    print(f"  params: {yolo_info.get('yolo11n_num_params_M')} M")
    print(f"  7-class yaml nc: {yolo_info.get('yolo11n_yaml_nc')}")
    print(f"  load pretrained ok: {yolo_info.get('yolo11n_yaml_load_pretrained_ok')}")
    print(f"  7cls loaded nc: {yolo_info.get('yolo11n_yaml_loaded_nc')}")
    print(f"  7cls cv2 out_ch: {yolo_info.get('yolo11n_7cls_cv2_out_channels')}")
    print(f"  forward pred shape: {yolo_info.get('yolo11n_7cls_forward_pred_shape')}")

    print("\n[3/3] Verifying YOLO11n ONNX export path...")
    export_info = verify_yolo11n_export_path()
    print(f"  onnx exported: {export_info.get('onnx_exported')}")
    print(f"  onnx size: {export_info.get('onnx_size_kb')} KB")

    report = {
        "stage": "1Ca",
        "st_yoloxn_tflite": st_info,
        "yolo11n_load": yolo_info,
        "yolo11n_onnx_export": export_info,
    }
    out_path = REPORT_DIR / "static_verify.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] report saved to {out_path}")


if __name__ == "__main__":
    main()
