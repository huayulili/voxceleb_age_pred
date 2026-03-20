# VoxCeleb Age Prediction

基于 `voice_age_spec_v3.md` 的 AgeVoxCeleb 年龄预测训练工程。

## 用法

```bash
python train.py --mode ldl_bc --output_dir ./outputs/E6
python evaluate.py --checkpoint ./outputs/E6/best_model.pt --save_plots
```

## 默认数据路径

- Age 标签：`/home/bld/data1_disk/daavee/dataset/voice_dataset/voxceleb_enrichment_age_gender/dataset`
- VoxCeleb 原始语音：`/data1/daavee/dataset/voice_dataset/voxceleb`

## 注意

- 当前实现优先支持“已解压目录”或“完整 zip 包”。
- 若只有 `vox1_dev_wav_partaa` / `vox2_dev_aac_partaa` 这类分卷包，请先自行合并或解压后再训练。
- `--sex_specific` 在缺少性别 metadata 时会自动降级为统一模型。
