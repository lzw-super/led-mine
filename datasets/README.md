# Symlink/Put all the datasets here

It is recommended to symlink your dataset root to this folder - `datasets` with the command `ln -s xxx yyy`.

For the shared SID layout used by the quick-operation scripts, keep both the original
Sony data and the generated `Sony_npy` / `Sony_train_long_patches` under
`/home/shared_files/dataset/SID/`, then create exactly one repository link:

```bash
ln -s /home/shared_files/dataset/SID datasets/ICCV23-LED
```

## Txt files for benchmark and EMoR data for visualization

We have added the txt files for benchmark (the data list file for both SID and ELD dataset) in commit [`fadffc7`](https://github.com/Srameo/LED/commit/fadffc7282b02ab2fcc7fbade65f87217b642588).<br/>
Also, added the calibrated EMoR data in [ELD](https://github.com/Vandermode/ELD) for visualization on CUDA.
