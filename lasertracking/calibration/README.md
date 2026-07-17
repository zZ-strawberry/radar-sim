# 激光-相机标定工具

本目录包含两个标定工具：

- **`boresight_calibrator.py`** — 单参考距离同轴校准，输出 `boresight.yaml`
- **`parallax_estimator.py`** — 分析不同距离下基线视差带来的像素残差

主程序运行时瞄准点计算链路：`boresight 固定偏移` + `parallax 距离残差` → 最终瞄准点。

---

## 1. boresight vs parallax

| 误差类型 | 来源 | 处理方式 |
|---------|------|---------|
| 固定误差 | 安装角偏差、机械偏移 | boresight 校准（一次） |
| 距离误差 | 激光出光点与光心不重合 | 运行时 parallax 补偿，残差 ∝ 1/Z |

推荐流程：在一个参考距离做 boresight → 测量基线 bx/by → 启用运行时 parallax。

无实时测距时，可将 `parallax_runtime.source` 设为 `fixed`，取作用距离区间中点。

---

## 2. boresight_calibrator.py

### 常规模式（YOLO 辅助）

```
python lasertracking/calibration/boresight_calibrator.py --config lasertracking/tracking_system_old/config_tracking.yaml
```

自动将十字对准检测框中心，再手动微调至激光落点。

### 手动模式（无检测模型）

```
python lasertracking/calibration/boresight_calibrator.py --config lasertracking/tracking_system_old/config_tracking.yaml --no-detector
```

纯手工点击激光落点完成校准。

### 参数

| 参数 | 说明 |
|------|------|
| `--config` | 主配置文件路径 |
| `--output` | boresight 输出路径 |
| `--no-detector` | 不加载 YOLO，纯手工模式 |
| `--no-sync-config` | 保存时不自动同步 pixel_offset 回配置文件 |
| `--check-config` | 仅检查配置与路径，不打开相机 |
| `--step` | 初始微调步长（像素） |
| `--window-size` | 预览窗口尺寸 |

### 交互按键

| 按键 | 功能 |
|------|------|
| 方向键 / WASD | 移动十字 |
| `[` `-` / `]` `+` `=` | 减小 / 增大步长 |
| `c` | 十字对齐检测框中心 |
| `i` | 设当前位置为初值 |
| `r` | 回到初值 |
| `s` | 保存 boresight.yaml |
| `p` | 切换 parallax overlay |
| `q` / ESC | 退出 |
| 鼠标左键 | 十字移至点击位置 |
| 鼠标右键 | 设当前位置为初值 |
| 鼠标滚轮 | 调整步长 |

---

## 3. boresight.yaml 输出内容

| 字段 | 说明 |
|------|------|
| `u_L` / `v_L` | 激光参考落点像素坐标 |
| `image_w` / `image_h` | 标定时图像分辨率 |
| `bx` / `by` | 激光器相对相机光心基线 (m) |
| `z_ref` | 标定参考距离 (m) |

加载时按当前分辨率重映射后转换为相对画面中心的偏移量。

---

## 4. parallax_estimator.py

### 绝对偏移分析

```
python lasertracking/calibration/parallax_estimator.py --config lasertracking/calibration/parallax.yaml
```

输出 `du(Z) = fx·bx/Z`、`dv(Z) = fy·by/Z`。

### 相对参考距离分析

```
python lasertracking/calibration/parallax_estimator.py --config lasertracking/calibration/parallax.yaml --relative
```

输出 `du_res(Z) = du(Z) - du(z_ref)`，与主程序在线补偿行为一致。

### 导出 CSV

```
python lasertracking/calibration/parallax_estimator.py --config lasertracking/calibration/parallax.yaml --relative --csv outputs/parallax.csv
```

---

## 5. 运行时 parallax 配置

配置位置：`lasertracking/tracking_system_old/config_tracking.yaml` → `calibration.parallax_runtime`

```yaml
calibration:
  baseline:
    bx: 0.02
    by: 0.0
    z_ref: 30.0
  parallax_runtime:
    enabled: true
    source: auto
    fixed_distance_m: 20.0
    target_width_m: 0.0
    target_height_m: 0.0
    min_distance_m: 10.0
    max_distance_m: 40.0
    distance_alpha: 0.35
    hold_last_distance: true
    min_bbox_px: 8.0
```

| 字段 | 说明 |
|------|------|
| `enabled` | 启用运行时视差补偿 |
| `source` | 距离来源：`fixed` / `external` / `bbox_width` / `bbox_height` / `bbox_auto` / `auto` |
| `fixed_distance_m` | 固定距离 (m)，`fixed` 模式或 `auto` 无其他来源时的回退值 |
| `target_width_m` / `target_height_m` | 目标物理尺寸 (m)，用于 bbox 估距 |
| `min_distance_m` / `max_distance_m` | 距离裁剪范围 |
| `distance_alpha` | 距离一阶低通滤波系数 (0~1) |
| `hold_last_distance` | 短时丢检时沿用上次有效距离 |
| `min_bbox_px` | 检测框小于此值时不触发 bbox 估距 |

### source 选择指南

| 模式 | 适用场景 |
|------|---------|
| `fixed` | 距离变化小，无测距源 |
| `external` | 有外部测距输入（激光/雷达/双目） |
| `bbox_width` / `bbox_height` | 目标物理尺寸稳定，检测框可靠 |
| `bbox_auto` | 自动取宽、高估距的均值 |
| `auto` | 优先 bbox，回退 fixed |

> bbox 估距公式：`Z = fx · W_real / W_px`，受目标姿态影响较大，适合作为工程近似。

---

## 6. 主程序运行时行为

- 优先加载 `boresight.yaml`，其次回退 `pixel_offset`
- 读取 `baseline.bx/by/z_ref`
- 按 `source` 获取距离 → 一阶低通平滑 → 计算相对 z_ref 的视差残差
- 最终瞄准偏移 = `boresight 固定偏移 ± parallax 距离残差`
