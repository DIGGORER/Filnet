This code includes three projects: FillNet_mask_loss, FillNet_mse_loss, and FillNet_ssim_loss, with the main program for each located in Isonet/bin/main.py.
The main.py file contains five core methods: prepare_star, deconv, make_mask, refine, and predict.
The prepare_star method is mainly used to load dataset paths.
The deconv method performs CTF deconvolution to enhance image quality.
The make_mask method generates masks to locate valid information.

Both the deconv and make_mask methods are optional and do not affect the execution of subsequent processes.
The refine method is used for model training, while the predict method enables the restoration of cryo-electron microscopy images.
