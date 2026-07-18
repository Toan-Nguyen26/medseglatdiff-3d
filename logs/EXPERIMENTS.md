Master index of training runs. One line per run, appended automatically via `RunLogger.append_to_experiments_index(...)`. Keep this file committed to git — it's the cheap, scannable history; the heavy artifacts (checkpoints, full metric CSVs) live ungit-tracked under `logs/runs/<run_id>/` and should be synced to external storage (Kaggle Datasets / cloud bucket / HF Hub).

| Date | Run ID | Stage | Summary |
|---|---|---|---|
| 2026-06-29 | stage3_diffusion_frozen_20260629_111439 | stage3_diffusion_frozen | Tiny diffusion U-Net, 4 steps, final_loss=1.0130 |
| 2026-06-29 | stage3_diffusion_frozen_20260629_111507 | stage3_diffusion_frozen | Tiny diffusion U-Net, 6 steps, final_loss=1.0135 |
| 2026-07-01 | joint_20260701_093357 | joint | 4-encoder joint, 4 steps, embed_dim=64, final_loss=0.9815 |
| 2026-07-02 | pixel_test_20260702_103658 | pixel_test | pixel-space test, 4 steps, embed_dim=8, final_loss=1.4162 |
| 2026-07-02 | pixel_test_20260702_103901 | pixel_test | pixel-space test, 2 steps, embed_dim=8, final_loss=1.4373 |
| 2026-07-02 | pixel_test_20260702_104129 | pixel_test | pixel-space test, 100 steps, embed_dim=256, final_loss=0.0401 |
| 2026-07-02 | pixel_test_20260702_142304 | pixel_test | pixel-space test, 200 steps, embed_dim=256, final_loss=0.0584 |
| 2026-07-02 | pixel_test_20260702_155333 | pixel_test | pixel-space test, 1 epochs (4 steps), embed_dim=8, final_loss=0.7285 |
| 2026-07-02 | pixel_test_20260702_155354 | pixel_test | pixel-space test, 100 epochs (2500 steps), embed_dim=256, final_loss=0.0082 |
| 2026-07-03 | image_vae_20260703_185451 | image_vae | Image VAE, 2 epochs (100 steps), embed_dim=256, beta=0.0001, final_recon=0.1405 |
| 2026-07-03 | image_vae_20260703_190302 | image_vae | Image VAE, 2 epochs (100 steps), embed_dim=256, beta=0.0001, final_recon=0.1848 |
| 2026-07-03 | image_vae_20260703_203550 | image_vae | Image VAE, 2 epochs (88 steps), channels=64,128,256,256, res_units=2, beta=0.0001, final_recon=0.0686 |
| 2026-07-03 | image_vae_20260703_203948 | image_vae | Image VAE, 2 epochs (88 steps), channels=64,128,256,256, res_units=2, beta=0.0001, final_recon=0.0686 |
| 2026-07-03 | image_vae_20260703_204828 | image_vae | Image VAE, 50 epochs (2150 steps), channels=64,128,256,256, res_units=2, beta=0.0001, final_recon=0.0469 |
| 2026-07-04 | pixel_test_20260704_104014 | pixel_test | pixel-space test, 30 epochs (2610 steps), channels=64,128,256,256, final_loss=0.7717 |
| 2026-07-05 | pixel_test_20260705_202942 | pixel_test | pixel-space test, 50 epochs (4350 steps), channels=64,128,256,256, final_loss=0.0128 |
| 2026-07-06 | mask_vae_20260706_124416 | mask_vae | Mask VAE, 50 epochs (4350 steps), latent=4ch, channels=32,64,128,128, beta=0.01, final_recon=0.2850 |
| 2026-07-06 | diffusion_20260706_181017 | diffusion | pixel diffusion, 3 epochs (264 steps), base_ch=32, final_loss=0.0927 |
| 2026-07-14 | mask_vae_20260714_102925 | mask_vae | Mask VAE, 50 epochs (4350 steps), latent=4ch, channels=32,64,128,128, beta=0.01, final_recon=0.1376 |
| 2026-07-16 | mask_vae_20260715_230345 | mask_vae | Mask VAE, 50 epochs (4350 steps), latent=4ch, channels=32,64,128, beta=0.01, final_recon=0.1009 |
| 2026-07-17 | latent_diffusion_20260717_223755 | latent_diffusion | latent diffusion, 10 steps, best_dice=0.0438, unet=2.2M, embed=256, subregion=False |
| 2026-07-17 | latent_diffusion_20260717_223943 | latent_diffusion | latent diffusion, 100 steps, best_dice=0.0500, unet=2.2M, embed=256, subregion=False |
