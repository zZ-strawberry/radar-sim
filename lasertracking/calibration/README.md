# 激光相机标定工具说明

这个目录下放的是激光器与相机配准相关的小工具，职责分成两层：

- `boresight_calibrator.py`
  用来做单参考距离的同轴校准，得到 `boresight.yaml`。
- `parallax_estimator.py`
  用来分析“固定基线 + 近似平行光轴”条件下，不同距离带来的像素残差。

主程序 `lasertracking/aerial_tracking_system.py` 现在已经接入了运行时视差补偿链路：

- 基础瞄准点使用 `boresight.yaml` 或旧的 `pixel_offset`
- 距离相关残差按 `du = fx * bx / Z`、`dv = fy * by / Z` 计算
- 若设置了 `z_ref`，运行时补的是相对 `z_ref` 的残差
- 最终在线瞄准点 = `boresight` 固定偏移 + `parallax` 距离残差

## 1. 什么时候只做 boresight，什么时候还要 parallax

如果激光器和相机位置固定，光轴也基本平行，那么不需要“每个距离都重新人工标一次”。

你真正要处理的是两部分误差：

- 固定误差：安装角偏差、固定机械偏移
- 距离误差：激光出光点与相机光心不重合产生的视差，近似随 `1 / Z` 变化

所以推荐做法是：

1. 在一个参考距离 `z_ref` 做一次高质量 boresight
2. 量出 `bx/by`
3. 在主程序里启用运行时视差补偿

如果你暂时没有实时距离来源，也可以先只做单距离 boresight，然后把 `parallax_runtime.source` 设成 `fixed` 或 `auto + fixed_distance_m`。  
对于 15m 到 28m 这种区间，固定距离先取中间值附近通常比只取最远距离更稳。

## 2. `boresight_calibrator.py` 用法

### 2.1 常规模式

```powershell
python lasertracking/calibration/boresight_calibrator.py --config lasertracking/config_tracking.yaml
```

适合场景：

- 画面里能检测到目标
- 希望先把十字快速吸到检测框中心，再手工微调到真实激光点

### 2.2 手动模式

```powershell
python lasertracking/calibration/boresight_calibrator.py --config lasertracking/config_tracking.yaml --no-detector
```

适合场景：

- 没有可用检测模型
- 只想手工点击激光落点做校准

### 2.3 常用参数

```powershell
python lasertracking/calibration/boresight_calibrator.py `
  --config lasertracking/config_tracking.yaml `
  --output lasertracking/calibration/boresight.yaml `
  --step 2 `
  --window-size 1280x720
```

参数说明：

- `--config`
  主配置文件路径，默认应指向 `lasertracking/config_tracking.yaml`
- `--output`
  boresight 输出路径
- `--no-detector`
  不加载 YOLO，纯手工模式
- `--no-sync-config`
  保存后不自动同步 `pixel_offset` 回配置文件
- `--check-config`
  只检查配置和路径，不打开相机
- `--step`
  初始微调步长，单位像素
- `--window-size`
  预览窗口大小

### 2.4 交互按键

- 方向键 / `WASD`
  移动红色十字
- `[` / `-`
  减小步长
- `]` / `+` / `=`
  增大步长
- `c`
  将十字对齐到检测框中心
- `i`
  将当前位置设为初值
- `r`
  回到初值
- `s`
  保存 `boresight.yaml`
- `p`
  开关 `parallax overlay`
- `q` / `ESC`
  退出
- 鼠标左键
  将十字移动到点击位置
- 鼠标右键
  将当前位置设为初值
- 鼠标滚轮
  调整步长

## 3. `boresight.yaml` 会保存什么

保存文件里通常包含：

- `u_L` / `v_L`
  激光参考落点在图像中的像素坐标
- `image_w` / `image_h`
  保存标定时的图像尺寸
- `bx` / `by`
  激光器相对相机光心的基线
- `z_ref`
  该次 boresight 的参考距离

主程序加载 `boresight.yaml` 时，会先按当前相机分辨率重映射，再转换成相对画面中心的偏移。

## 4. `parallax_estimator.py` 用法

### 4.1 绝对偏置分析

```powershell
python lasertracking/calibration/parallax_estimator.py --config lasertracking/calibration/parallax.yaml
```

输出的是不同距离下的绝对像素偏移：

- `du(Z) = fx * bx / Z`
- `dv(Z) = fy * by / Z`

### 4.2 相对参考距离分析

```powershell
python lasertracking/calibration/parallax_estimator.py --config lasertracking/calibration/parallax.yaml --relative
```

输出的是相对 `z_ref` 的残差：

- `du_res(Z) = du(Z) - du(z_ref)`
- `dv_res(Z) = dv(Z) - dv(z_ref)`

这个模式更贴近主程序在线补偿的实际含义。

### 4.3 导出 CSV

```powershell
python lasertracking/calibration/parallax_estimator.py `
  --config lasertracking/calibration/parallax.yaml `
  --relative `
  --csv outputs/parallax.csv
```

## 5. 主程序运行时补偿配置

配置位置在 `lasertracking/config_tracking.yaml` 的 `calibration.parallax_runtime`。

示例：

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

字段说明：

- `enabled`
  是否启用运行时视差补偿
- `source`
  距离来源，支持：
  - `auto`
  - `external`
  - `fixed`
  - `bbox_width`
  - `bbox_height`
  - `bbox_auto`
- `fixed_distance_m`
  固定距离模式，或 `auto` 没有其它来源时的回退值
- `target_width_m`
  目标真实宽度，给 `bbox_width` 或 `auto` 使用
- `target_height_m`
  目标真实高度，给 `bbox_height` 或 `auto` 使用
- `min_distance_m` / `max_distance_m`
  距离裁剪范围，防止异常值把补偿拉飞
- `distance_alpha`
  距离一阶低通滤波系数
- `hold_last_distance`
  短时丢检时是否沿用上一次有效距离
- `min_bbox_px`
  框太小时不做 `bbox` 估距

## 6. `source` 该怎么选

### 6.1 `fixed`

适合：

- 现场距离变化不大
- 没有测距源
- 先想把系统跑稳

对于 15m 到 28m 的范围，先把 `fixed_distance_m` 设在中间值附近通常比设在最远端更稳。

### 6.2 `external`

适合：

- 你后面会接入激光测距、双目、雷达或别的外部距离输入

主程序里已经预留了 `set_target_distance_m()` 接口，外部模块可以直接把距离灌进去。

### 6.3 `bbox_width` / `bbox_height` / `bbox_auto`

适合：

- 目标真实物理尺寸比较稳定
- 检测框质量足够好

估距近似是针孔模型：

- `Z = fx * W_real / W_px`
- `Z = fy * H_real / H_px`

注意：

- 如果目标姿态变化很大，`bbox` 尺寸法会抖
- 它适合作为“没有测距设备时的工程近似”，不是高精度真值

## 7. 实操建议

对于你现在这种“激光和相机固定，作用距离约 15m 到 28m”的场景，建议：

1. 先在 `19m` 到 `20m` 左右做一次高质量 boresight
2. 把 `bx/by` 量准
3. 主程序先开 `parallax_runtime.enabled: true`
4. 如果没有实时测距，先用 `source: auto` + `fixed_distance_m: 20.0`
5. 以后一旦有测距源，再切到 `external`

## 8. 当前主程序已实现的行为

现在主程序在线上会做这些事：

- 加载 `boresight.yaml`，优先于旧的 `pixel_offset`
- 读取 `baseline.bx/by/z_ref`
- 根据 `parallax_runtime.source` 选择距离来源
- 计算相对 `z_ref` 的视差残差
- 用 `base_aim_pixel_offset + parallax_residual` 生成最终瞄准点
- 在 UI 上显示当前距离来源、距离值、`parallax` 像素补偿量

                                                   
                                                                                                                   
python lasertracking/calibration/boresight_calibrator.py --config lasertracking/      config_tracking.yaml --no-detector                                                       
                                                                                                  
